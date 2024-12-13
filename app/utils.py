import os
from flask import current_app
from openai import OpenAI
from pydub import AudioSegment
import tempfile
from datetime import datetime
from werkzeug.utils import secure_filename
import math
import asyncio
import aiohttp
from typing import List
from asyncio import Semaphore
import time

MAX_CONCURRENT_REQUESTS = 3  # Reduced from 5
TRANSLATION_CHUNK_SIZE = 300  # Reduced from 500

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

def save_transcript(original_filename: str, transcript: str, chinese_transcript: str = None) -> str:
    """Save transcript to a markdown file with optional Chinese translation"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(original_filename)[0]
    safe_name = secure_filename(f"{base_name}_{timestamp}.md")
    file_path = os.path.join(current_app.config['TRANSCRIPTS_DIR'], safe_name)
    
    markdown_content = f"""# Transcript: {original_filename}
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## English Content

{transcript}
"""

    if chinese_transcript:
        markdown_content += f"""

## Chinese Content / 中文内容

{chinese_transcript}
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

async def process_chunk_async(client, chunk: str, chunk_num: int) -> tuple[int, str]:
    """Process a single chunk asynchronously"""
    try:
        print(f"Processing chunk {chunk_num}")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """You are a transcript editor. Your task is to:
1. Format the text into proper paragraphs
2. Add appropriate punctuation
3. Fix obvious transcription errors
4. Add markdown formatting where appropriate
5. Add section headings when the topic changes significantly
Do not add any commentary or change the meaning of the text."""},
                {"role": "user", "content": f"Please format this transcript chunk:\n\n{chunk}"}
            ],
            temperature=0.3
        )
        return chunk_num, response.choices[0].message.content
    except Exception as e:
        print(f"Error processing chunk {chunk_num}: {e}")
        return chunk_num, chunk  # Return original chunk if processing fails

async def process_chunks_concurrently(client, chunks: List[str]) -> List[str]:
    """Process multiple chunks concurrently"""
    tasks = []
    for i, chunk in enumerate(chunks):
        task = process_chunk_async(client, chunk, i)
        tasks.append(task)
    
    # Wait for all chunks to be processed
    results = await asyncio.gather(*tasks)
    
    # Sort results by chunk number and extract processed text
    sorted_results = sorted(results, key=lambda x: x[0])
    return [result[1] for result in sorted_results]

def process_text_with_gpt4(client, text: str) -> str:
    """Process text with GPT-4 to improve formatting and readability"""
    try:
        # Split text into chunks of approximately 2000 words
        words = text.split()
        chunks = []
        chunk_size = 2000
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        print(f"Split text into {len(chunks)} chunks")
        
        # Process chunks concurrently
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        processed_chunks = loop.run_until_complete(
            process_chunks_concurrently(client, chunks)
        )
        loop.close()
        
        # Combine processed chunks with section breaks
        return "\n\n---\n\n".join(processed_chunks)
        
    except Exception as e:
        print(f"Error processing text with GPT-4o: {e}")
        return text  # Return original text if processing fails

async def translate_chunk_async(client, chunk: str, chunk_num: int, semaphore: Semaphore) -> tuple[int, str]:
    """Translate a single chunk to Chinese asynchronously"""
    async with semaphore:  # Limit concurrent API calls
        try:
            print(f"Translating chunk {chunk_num}")
            start_time = time.time()
            
            # Use regular create method in an async context
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": """You are a professional translator. 
Translate English to Chinese while:
1. Maintaining the original meaning accurately
2. Using natural and fluent Chinese expressions
3. Preserving any markdown formatting
4. Keeping section headings in both English and Chinese"""},
                    {"role": "user", "content": f"Translate to Chinese:\n\n{chunk}"}
                ],
                temperature=0.3
            )
            
            duration = time.time() - start_time
            print(f"Chunk {chunk_num} translated in {duration:.2f}s")
            return chunk_num, response.choices[0].message.content
            
        except Exception as e:
            print(f"Error translating chunk {chunk_num}: {e}")
            return chunk_num, chunk

def split_into_sentence_chunks(text: str, max_chunk_size: int = TRANSLATION_CHUNK_SIZE) -> List[str]:
    """Split text into chunks at sentence boundaries"""
    sentences = text.split('.')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence = sentence + '.'
        sentence_size = len(sentence.split())
        
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

async def translate_text_concurrently(client, text: str) -> str:
    """Translate text to Chinese with concurrent processing"""
    try:
        # Split into smaller chunks at sentence boundaries
        chunks = split_into_sentence_chunks(text)
        total_chunks = len(chunks)
        print(f"Split text into {total_chunks} chunks for translation")
        
        # Create semaphore for rate limiting
        semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
        
        # Process chunks concurrently with rate limiting
        start_time = time.time()
        
        # Create and gather tasks in smaller batches
        all_results = []
        batch_size = MAX_CONCURRENT_REQUESTS
        
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            tasks = [
                translate_chunk_async(client, chunk, j + i, semaphore)
                for j, chunk in enumerate(batch_chunks)
            ]
            batch_results = await asyncio.gather(*tasks)
            all_results.extend(batch_results)
            
            # Add a small delay between batches to avoid rate limits
            if i + batch_size < total_chunks:
                await asyncio.sleep(1)
        
        # Sort results by chunk number and extract translated text
        sorted_results = sorted(all_results, key=lambda x: x[0])
        translated_text = "\n".join(result[1] for result in sorted_results)
        
        duration = time.time() - start_time
        print(f"Translated {total_chunks} chunks in {duration:.2f}s "
              f"({duration/total_chunks:.2f}s per chunk)")
        
        return translated_text
        
    except Exception as e:
        print(f"Error in translation: {e}")
        return text

def save_bilingual_transcript(original_filename: str, en_transcript: str, zh_transcript: str) -> str:
    """Save both English and Chinese transcripts to a markdown file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(original_filename)[0]
    safe_name = secure_filename(f"{base_name}_bilingual_{timestamp}.md")
    file_path = os.path.join(current_app.config['TRANSCRIPTS_DIR'], safe_name)
    
    markdown_content = f"""# Bilingual Transcript: {original_filename}

## Metadata
- **Source File:** {original_filename}
- **Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Duration:** [Duration will be added here]

## English Content

{en_transcript}

## Chinese Content / 中文内容

{zh_transcript}
"""
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    return file_path