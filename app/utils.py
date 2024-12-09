import os
from flask import current_app
from openai import OpenAI
from pydub import AudioSegment
import tempfile
from datetime import datetime
from werkzeug.utils import secure_filename
import math

def get_openai_client():
    """Create and return an OpenAI client instance"""
    api_key = current_app.config['OPENAI_API_KEY']
    if not api_key:
        raise ValueError("OpenAI API key not found in configuration")
    return OpenAI(api_key=api_key)

def process_large_audio(audio_path, user_session):
    """Process large audio files by splitting them into chunks"""
    try:
        print(f"Starting to process audio file: {audio_path}")
        audio = AudioSegment.from_file(audio_path)
        
        user_temp_dir = os.path.join(current_app.config['TEMP_DIR'], user_session)
        
        # Get file size
        file_size = os.path.getsize(audio_path)
        print(f"File size: {file_size / (1024*1024):.2f}MB")
        
        if file_size <= 10 * 1024 * 1024:  # 10MB
            print("File is small enough, no need to split")
            return [audio_path]
        
        # Calculate number of chunks needed
        total_duration = len(audio)
        chunk_duration = 10 * 60 * 1000  # 10 minutes in milliseconds
        num_chunks = math.ceil(total_duration / chunk_duration)
        print(f"Need to split into {num_chunks} chunks")
        
        chunk_paths = []
        
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = min((i + 1) * chunk_duration, total_duration)
            
            # Extract chunk
            chunk = audio[start_time:end_time]
            
            # Create temporary file for chunk
            chunk_path = os.path.join(
                user_temp_dir,
                f"chunk_{i}_{next(tempfile._get_candidate_names())}.mp3"
            )
            
            # Export chunk with compression
            chunk.export(
                chunk_path,
                format="mp3",
                parameters=["-q:a", "1"]
            )
            
            chunk_paths.append(chunk_path)
        
        return chunk_paths
    except Exception as e:
        print(f"Error in process_large_audio: {str(e)}")
        raise

def transcribe_audio_file(client, audio_path):
    """Transcribe a single audio file"""
    try:
        print(f"Starting transcription of: {audio_path}")
        with open(audio_path, 'rb') as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return transcription
    except Exception as e:
        print(f"Error in transcribe_audio_file: {str(e)}")
        raise

def save_transcript(original_filename: str, transcript: str) -> str:
    """Save transcript to a markdown file and return the file path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(original_filename)[0]
    safe_name = secure_filename(f"{base_name}_{timestamp}.md")
    file_path = os.path.join(current_app.config['TRANSCRIPTS_DIR'], safe_name)
    
    markdown_content = f"""# Transcript: {original_filename}
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Content

{transcript}
"""
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    return file_path

def cleanup_user_files(user_session):
    """Clean up all temporary files for a user session"""
    if not user_session:
        return
        
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