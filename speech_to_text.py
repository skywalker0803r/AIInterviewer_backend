import whisper
import logging
import tempfile
import time

whisper_model = None

async def load_whisper_model():
    global whisper_model
    logging.info("Loading Whisper model...")
    import torch
    if torch.cuda.is_available():
        logging.info("CUDA is available. Using GPU for Whisper.")
        whisper_model = whisper.load_model("turbo", device="cuda")
    else:
        logging.info("CUDA is not available. Using CPU for Whisper.")
        whisper_model = whisper.load_model("turbo", device="cpu")
    logging.info("Whisper model loaded.")

async def transcribe_audio(audio_chunk: bytes) -> str:
    transcribed_text = ""
    logging.info(f"Received candidate audio, size: {len(audio_chunk)} bytes.")
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmpfile:
        tmpfile.write(audio_chunk)
        tmpfile_path = tmpfile.name
        logging.info(f"Audio temporary file path: {tmpfile_path}")

        try:
            logging.info("Starting transcription...")
            start_time = time.time()
            result = whisper_model.transcribe(tmpfile_path, language="zh")
            end_time = time.time()
            transcribed_text = result["text"]
            logging.info(f"Transcription successful. Time taken: {end_time - start_time:.2f} seconds. Whisper result: {transcribed_text}")
        except Exception as e:
            logging.error(f"Whisper speech-to-text failed: {e}. Full error: {e}")
    return transcribed_text
