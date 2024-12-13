import os
import asyncio
from typing import List
from app.celery_app import celery
from app.utils import (
    get_openai_client,
    process_large_audio,
    transcribe_audio_file,
    save_transcript,
    cleanup_user_files,
    process_text_with_gpt4
)
from celery import group
from datetime import datetime

async def transcribe_chunk_async(client, chunk_path: str, chunk_num: int) -> tuple[int, str]:
    """Transcribe a single audio chunk asynchronously"""
    try:
        print(f"Transcribing chunk {chunk_num}: {chunk_path}")
        with open(chunk_path, 'rb') as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return chunk_num, transcription
    except Exception as e:
        print(f"Error transcribing chunk {chunk_num}: {e}")
        raise

async def transcribe_chunks_concurrently(client, chunk_paths: List[str]) -> List[str]:
    """Transcribe multiple chunks concurrently"""
    tasks = []
    for i, chunk_path in enumerate(chunk_paths):
        task = transcribe_chunk_async(client, chunk_path, i)
        tasks.append(task)
    
    # Wait for all chunks to be transcribed
    results = await asyncio.gather(*tasks)
    
    # Sort results by chunk number and extract transcribed text
    sorted_results = sorted(results, key=lambda x: x[0])
    return [result[1] for result in sorted_results]

@celery.task
def process_transcription(file_path, user_session, filename):
    try:
        print(f"Starting transcription task for file: {filename}")
        # Process audio file into chunks
        chunk_paths = process_large_audio(file_path, user_session)
        print(f"Created {len(chunk_paths)} chunks")
        
        # Get OpenAI client
        client = get_openai_client()
        
        # Transcribe all chunks concurrently
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        transcriptions = loop.run_until_complete(
            transcribe_chunks_concurrently(client, chunk_paths)
        )
        loop.close()
        
        # Combine transcriptions
        full_transcription = ' '.join(transcriptions)
        print("Transcription completed")
        
        # Save transcript
        transcript_path = save_transcript(filename, full_transcription)
        print(f"Saved transcript to: {transcript_path}")
        
        return {
            'status': 'success',
            'transcription': full_transcription,
            'saved_to': os.path.basename(transcript_path)
        }
    except Exception as e:
        print(f"Task error: {str(e)}")
        raise
    finally:
        cleanup_user_files(user_session)

@celery.task
def translate_chunk_task(chunk: str, chunk_num: int) -> tuple[int, str]:
    """Translate a single chunk as a Celery task"""
    try:
        print(f"Translating chunk {chunk_num}")
        client = get_openai_client()
        
        # Use synchronous API call
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
        
        result = (chunk_num, response.choices[0].message.content)
        print(f"Chunk {chunk_num} translated")
        return result
        
    except Exception as e:
        print(f"Error translating chunk {chunk_num}: {e}")
        return chunk_num, chunk

@celery.task
def combine_translations(results):
    """Combine translated chunks in correct order"""
    try:
        # Sort results by chunk number
        sorted_results = sorted(results, key=lambda x: x[0])
        translated_text = "\n".join(result[1] for result in sorted_results)
        print(f"Combined {len(results)} translations")
        return translated_text
    except Exception as e:
        print(f"Error combining translations: {e}")
        return ""

@celery.task
def save_translation_task(translation: str, filename: str):
    """Save translation and return the translated text"""
    try:
        # Get the transcripts directory from config
        from app import create_app
        app = create_app()
        
        with app.app_context():
            # Read existing file
            file_path = os.path.join(app.config['TRANSCRIPTS_DIR'], filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get English content
            english_content = content.split('## Chinese Content')[0].strip()
            
            # Create new content with both versions
            new_content = f"{english_content}\n\n## Chinese Content / 中文内容\n\n{translation}"
            
            # Save updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            return {'status': 'completed', 'translation': translation}
    except Exception as e:
        print(f"Error saving translation: {e}")
        return {'status': 'error', 'translation': translation}

@celery.task
def process_chunk_task(chunk: str, chunk_num: int) -> tuple[int, str]:
    """Process a single chunk as a Celery task"""
    try:
        print(f"Processing chunk {chunk_num}")
        client = get_openai_client()
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """You are a transcript editor. Your task is to:
1. Format the text into proper paragraphs
2. Add appropriate punctuation where needed
3. Fix obvious transcription errors
4. Keep the original meaning and all content intact
5. Do not add any commentary or additional content
6. Do not add section headings or markdown formatting
Just focus on making the text more readable with proper paragraphing."""},
                {"role": "user", "content": f"Format this transcript chunk into proper paragraphs:\n\n{chunk}"}
            ],
            temperature=0.3
        )
        
        result = (chunk_num, response.choices[0].message.content)
        print(f"Chunk {chunk_num} processed")
        return result
        
    except Exception as e:
        print(f"Error processing chunk {chunk_num}: {e}")
        return chunk_num, chunk

@celery.task
def combine_processed_chunks(results):
    """Combine processed chunks in correct order"""
    try:
        # Sort results by chunk number
        sorted_results = sorted(results, key=lambda x: x[0])
        processed_text = "\n\n".join(result[1] for result in sorted_results)
        print(f"Combined {len(results)} processed chunks")
        return processed_text
    except Exception as e:
        print(f"Error combining processed chunks: {e}")
        return ""

@celery.task
def save_processed_text(processed_text: str, filename: str):
    """Save processed text and return it"""
    try:
        # Get the transcripts directory from config
        from app import create_app
        app = create_app()
        
        with app.app_context():
            # Read existing file to preserve any translations
            file_path = os.path.join(app.config['TRANSCRIPTS_DIR'], filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if file has Chinese translation
            sections = content.split('## Chinese Content')
            if len(sections) > 1:
                chinese_content = sections[1]
                new_content = f"""# Transcript: {filename}
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## English Content

{processed_text}

## Chinese Content{chinese_content}"""
            else:
                new_content = f"""# Transcript: {filename}
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## English Content

{processed_text}"""
            
            # Save updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            return {'status': 'completed', 'processed_text': processed_text}
    except Exception as e:
        print(f"Error saving processed text: {e}")
        return {'status': 'error', 'processed_text': processed_text}