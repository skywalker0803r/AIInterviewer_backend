from gtts import gTTS
import io
import uuid
import logging
from gcs_utils import upload_audio_to_gcs
from config import GCS_BUCKET_NAME
import time

async def generate_and_upload_audio(text: str) -> str:
    logging.info(f"開始為文本生成音訊: '{text[:50]}...'")
    start_time = time.time()
    tts = gTTS(text, lang="zh-tw")
    audio_stream = io.BytesIO()
    tts.write_to_fp(audio_stream)
    audio_stream.seek(0)

    audio_filename = f"{uuid.uuid4().hex}.mp3"
    logging.info(f"音訊檔案名: {audio_filename}。開始上傳到 GCS...")
    upload_start_time = time.time()
    audio_url = await upload_audio_to_gcs(audio_stream.getvalue(), audio_filename, GCS_BUCKET_NAME)
    upload_end_time = time.time()
    logging.info(f"音訊已上傳到 GCS: {audio_url}，上傳耗時: {upload_end_time - upload_start_time:.2f} 秒。")
    
    end_time = time.time()
    logging.info(f"音訊生成和上傳完成，總耗時: {end_time - start_time:.2f} 秒。")
    return audio_url
