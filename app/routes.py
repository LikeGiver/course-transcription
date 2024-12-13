from flask import Blueprint, render_template, request, jsonify, current_app, session
from werkzeug.utils import secure_filename
import os
from openai import OpenAI
import tempfile
from moviepy.editor import VideoFileClip
from datetime import datetime
from pydub import AudioSegment
import math
from werkzeug.exceptions import RequestEntityTooLarge
import uuid
from app.tasks import (
    process_transcription, 
    translate_chunk_task, 
    combine_translations, 
    process_chunk_task, 
    combine_processed_chunks, 
    save_processed_text,
    save_translation_task
)
from app.utils import process_large_audio, transcribe_audio_file, save_transcript, cleanup_user_files, process_text_with_gpt4, translate_text_concurrently, split_into_sentence_chunks
import asyncio
from celery import group
from app.celery_app import celery

main = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {'mp4', 'mp3', 'wav', 'webm', 'mpga', 'm4a'}
MAX_CHUNK_SIZE = 10 * 1024 * 1024  # 24MB to be safe
CHUNK_DURATION = 10 * 60 * 1000  # 10 minutes in milliseconds

def ensure_directories():
    """Ensure all required directories exist"""
    os.makedirs(current_app.config['TEMP_DIR'], exist_ok=True)
    os.makedirs(current_app.config['TRANSCRIPTS_DIR'], exist_ok=True)

def get_openai_client():
    """Create and return an OpenAI client instance"""
    api_key = current_app.config['OPENAI_API_KEY']
    if not api_key:
        raise ValueError("OpenAI API key not found in configuration")
    return OpenAI(api_key=api_key)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_transcript(original_filename: str, transcript: str) -> str:
    """Save transcript to a markdown file and return the file path"""
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create safe filename
    base_name = os.path.splitext(original_filename)[0]
    safe_name = secure_filename(f"{base_name}_{timestamp}.md")
    
    # Create full path
    file_path = os.path.join(current_app.config['TRANSCRIPTS_DIR'], safe_name)
    
    # Create markdown content
    markdown_content = f"""# Transcript: {original_filename}
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Content

{transcript}
"""
    
    # Save to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    return file_path

def convert_video_to_audio(video_path):
    """Convert video file to audio file"""
    try:
        # Create a temporary file for the audio in our temp directory
        audio_path = os.path.join(
            current_app.config['TEMP_DIR'],
            next(tempfile._get_candidate_names()) + '.mp3'
        )
        
        # Load the video file
        video = VideoFileClip(video_path)
        # Extract the audio
        video.audio.write_audiofile(audio_path, codec='mp3')
        # Close the video to release resources
        video.close()
        
        return audio_path
    except Exception as e:
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        raise Exception(f"Error converting video to audio: {str(e)}")

def process_large_audio(audio_path, user_session):
    """Process large audio files by splitting them into chunks"""
    try:
        print(f"Starting to process audio file: {audio_path}")
        # Load the audio file
        audio = AudioSegment.from_file(audio_path)
        print(f"Successfully loaded audio file, duration: {len(audio)}ms")
        
        # Get file size
        file_size = os.path.getsize(audio_path)
        print(f"File size: {file_size / (1024*1024):.2f}MB")
        
        if file_size <= MAX_CHUNK_SIZE:
            print("File is small enough, no need to split")
            return [audio_path]
        
        # Calculate number of chunks needed
        total_duration = len(audio)
        num_chunks = math.ceil(total_duration / CHUNK_DURATION)
        print(f"Need to split into {num_chunks} chunks")
        
        chunk_paths = []
        
        user_temp_dir = os.path.join(current_app.config['TEMP_DIR'], user_session)
        
        for i in range(num_chunks):
            start_time = i * CHUNK_DURATION
            end_time = min((i + 1) * CHUNK_DURATION, total_duration)
            print(f"Processing chunk {i+1}/{num_chunks} ({start_time}ms to {end_time}ms)")
            
            # Extract chunk
            chunk = audio[start_time:end_time]
            
            # Create temporary file for chunk
            chunk_path = os.path.join(
                user_temp_dir,
                f"chunk_{i}_{next(tempfile._get_candidate_names())}.mp3"
            )
            print(f"Saving chunk to: {chunk_path}")
            
            # Export chunk with compression
            chunk.export(
                chunk_path,
                format="mp3",
                parameters=["-q:a", "1"]  # High quality compression
            )
            print(f"Successfully exported chunk {i+1}")
            
            chunk_paths.append(chunk_path)
        
        print(f"Successfully created {len(chunk_paths)} chunks")
        return chunk_paths
    
    except Exception as e:
        print(f"Error in process_large_audio: {str(e)}")
        raise Exception(f"Error processing large audio file: {str(e)}")

def transcribe_audio_file(client, audio_path):
    """Transcribe a single audio file and post-process with GPT-4"""
    try:
        print(f"Starting transcription of: {audio_path}")
        file_size = os.path.getsize(audio_path)
        print(f"File size to transcribe: {file_size / (1024*1024):.2f}MB")
        
        with open(audio_path, 'rb') as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        
        print("Raw transcription complete, starting post-processing")
        processed_text = process_text_with_gpt4(client, transcription)
        print("Post-processing complete")
        
        return processed_text
    except Exception as e:
        print(f"Error in transcribe_audio_file: {str(e)}")
        raise

def cleanup_user_files(user_session):
    """Clean up all temporary files for a user session"""
    user_temp_dir = os.path.join(current_app.config['TEMP_DIR'], user_session)
    try:
        if os.path.exists(user_temp_dir):
            for filename in os.listdir(user_temp_dir):
                file_path = os.path.join(user_temp_dir, filename)
                try:
                    os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
            os.rmdir(user_temp_dir)
    except Exception as e:
        print(f"Error cleaning up user directory: {e}")

@main.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@main.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'user_session' not in session:
            session['user_session'] = str(uuid.uuid4())
        
        user_session = session['user_session']
        user_temp_dir = os.path.join(current_app.config['TEMP_DIR'], user_session)
        os.makedirs(user_temp_dir, exist_ok=True)
        
        ensure_directories()
        print("Starting transcription process")
        
        if 'file' not in request.files:
            print("No file in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        print(f"Received file: {file.filename}")
        
        if file.filename == '':
            print("No filename")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            print(f"File type not allowed: {file.filename}")
            return jsonify({'error': 'File type not allowed'}), 400

        # Get file size before saving
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset file pointer
        print(f"Upload file size: {file_size / (1024*1024):.2f}MB")

        # Save file to user's temp directory
        temp_path = os.path.join(user_temp_dir, secure_filename(file.filename))
        file.save(temp_path)
        
        # Convert video to audio if needed
        if file.filename.lower().endswith(('.mp4', '.webm')):
            print("Converting video to audio")
            audio_path = convert_video_to_audio(temp_path)
            os.unlink(temp_path)
            temp_path = audio_path
            print(f"Video converted to audio: {audio_path}")

        # Start background task
        task = process_transcription.delay(temp_path, user_session, file.filename)
        
        return jsonify({
            'task_id': task.id,
            'status': 'processing'
        })
        
    except Exception as e:
        print(f"Error in transcribe route: {str(e)}")
        cleanup_user_files(session.get('user_session'))
        return jsonify({'error': str(e)}), 500

@main.route('/transcripts', methods=['GET'])
def get_transcripts():
    """Get list of all transcription files"""
    try:
        transcripts = []
        for filename in os.listdir(current_app.config['TRANSCRIPTS_DIR']):
            if filename.endswith('.md'):
                file_path = os.path.join(current_app.config['TRANSCRIPTS_DIR'], filename)
                # Get file creation time
                creation_time = os.path.getctime(file_path)
                creation_date = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
                
                # Get original filename (remove timestamp and .md)
                original_name = filename.rsplit('_', 1)[0]
                
                transcripts.append({
                    'filename': filename,
                    'original_name': original_name,
                    'created_at': creation_date
                })
        
        # Sort by creation time, newest first
        transcripts.sort(key=lambda x: x['created_at'], reverse=True)
        return jsonify(transcripts)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/transcripts/<filename>', methods=['GET'])
def get_transcript(filename):
    """Get content of a specific transcript"""
    try:
        file_path = os.path.join(current_app.config['TRANSCRIPTS_DIR'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'Transcript not found'}), 404
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return jsonify({'content': content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    try:
        print(f"Checking status for task: {task_id}")
        task = process_transcription.AsyncResult(task_id)
        
        if task.ready():
            result = task.get()
            print(f"Task completed with result: {result}")
            return jsonify(result)
            
        print("Task still processing")
        return jsonify({'status': 'processing'})
    except Exception as e:
        print(f"Error checking task status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@celery.task
def save_translation_task(translation: str, filename: str):
    """Save translation and return the translated text"""
    try:
        # Read existing file
        file_path = os.path.join(current_app.config['TRANSCRIPTS_DIR'], filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get English content
        english_content = content.split('## Chinese Content')[0].strip()
        
        # Create new content with both versions
        new_content = f"{english_content}\n\n## Chinese Content / 中文内容\n\n{translation}"
        
        # Save updated content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        # Return the translation for the frontend
        return translation
    except Exception as e:
        print(f"Error saving translation: {e}")
        return translation  # Still return translation even if save fails

@main.route('/translate', methods=['POST'])
def translate():
    try:
        text = request.json.get('text')
        filename = request.json.get('filename')
        if not text or not filename:
            return jsonify({'error': 'Missing text or filename'}), 400
        
        # Split text into chunks
        chunks = split_into_sentence_chunks(text)
        print(f"Split text into {len(chunks)} chunks for translation")
        
        # Create a group of tasks for parallel processing
        translation_tasks = group(
            translate_chunk_task.s(chunk, i)
            for i, chunk in enumerate(chunks)
        )
        
        # Execute tasks, combine results, and save
        result = (translation_tasks | combine_translations.s() | save_translation_task.s(filename))()
        
        return jsonify({
            'task_id': result.id,
            'status': 'processing'
        })
        
    except Exception as e:
        print(f"Translation error: {e}")
        return jsonify({'error': str(e)}), 500

@main.route('/translate/status/<task_id>', methods=['GET'])
def get_translation_status(task_id):
    try:
        result = celery.AsyncResult(task_id)
        
        if result.ready():
            result_data = result.get()
            if isinstance(result_data, dict):
                return jsonify(result_data)
            else:
                # Handle legacy format or direct translation text
                return jsonify({
                    'status': 'completed',
                    'translation': result_data
                })
        
        return jsonify({'status': 'processing'})
        
    except Exception as e:
        print(f"Error checking translation status: {e}")
        return jsonify({'error': str(e)}), 500

@main.route('/process', methods=['POST'])
def process_text():
    try:
        text = request.json.get('text')
        filename = request.json.get('filename')
        if not text or not filename:
            return jsonify({'error': 'Missing text or filename'}), 400
        
        # Split text into chunks
        chunks = split_into_sentence_chunks(text, max_chunk_size=500)  # Smaller chunks for better processing
        print(f"Split text into {len(chunks)} chunks for processing")
        
        # Create a group of tasks for parallel processing
        processing_tasks = group(
            process_chunk_task.s(chunk, i)
            for i, chunk in enumerate(chunks)
        )
        
        # Execute tasks, combine results, and save
        result = (processing_tasks | combine_processed_chunks.s() | save_processed_text.s(filename))()
        
        return jsonify({
            'task_id': result.id,
            'status': 'processing'
        })
        
    except Exception as e:
        print(f"Processing error: {e}")
        return jsonify({'error': str(e)}), 500

@main.route('/process/status/<task_id>', methods=['GET'])
def get_process_status(task_id):
    try:
        result = celery.AsyncResult(task_id)
        
        if result.ready():
            result_data = result.get()
            if isinstance(result_data, dict):
                return jsonify(result_data)
            else:
                return jsonify({
                    'status': 'completed',
                    'processed_text': result_data
                })
        
        return jsonify({'status': 'processing'})
        
    except Exception as e:
        print(f"Error checking process status: {e}")
        return jsonify({'error': str(e)}), 500