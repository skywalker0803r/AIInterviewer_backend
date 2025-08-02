from gtts import gTTS
import io
import uuid
import logging
from gcs_utils import upload_audio_to_gcs
from config import GCS_BUCKET_NAME

async def generate_and_upload_audio(text: str) -> str:
    tts = gTTS(text, lang="zh-TW")
    audio_stream = io.BytesIO()
    tts.write_to_fp(audio_stream)
    audio_stream.seek(0)

    audio_filename = f"{uuid.uuid4().hex}.mp3"
    audio_url = await upload_audio_to_gcs(audio_stream.getvalue(), audio_filename, GCS_BUCKET_NAME)
    logging.info(f"Generated and uploaded audio for text: {text[:30]}...")
    return audio_url
