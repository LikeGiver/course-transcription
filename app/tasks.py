import os
from app.celery_app import celery
from app.utils import (
    get_openai_client,
    process_large_audio,
    transcribe_audio_file,
    save_transcript,
    cleanup_user_files
)

@celery.task
def process_transcription(file_path, user_session, filename):
    try:
        print(f"Starting transcription task for file: {filename}")
        # Process audio file into chunks
        chunk_paths = process_large_audio(file_path, user_session)
        print(f"Created {len(chunk_paths)} chunks")
        
        # Get OpenAI client
        client = get_openai_client()
        
        # Transcribe all chunks
        transcriptions = []
        for i, chunk_path in enumerate(chunk_paths):
            print(f"Transcribing chunk {i+1}/{len(chunk_paths)}")
            chunk_transcription = transcribe_audio_file(client, chunk_path)
            transcriptions.append(chunk_transcription)
        
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