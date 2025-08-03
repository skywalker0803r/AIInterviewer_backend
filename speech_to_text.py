import whisper
import logging
import tempfile
import time

whisper_model = None

async def load_whisper_model():
    global whisper_model
    logging.info("開始載入 Whisper 模型...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"CUDA 是否可用: {torch.cuda.is_available()}。將使用 {device} 設備載入模型。")
    whisper_model = whisper.load_model("turbo", device=device)
    logging.info("Whisper 模型載入完成。")

from fastapi import UploadFile

async def transcribe_audio(audio_file: UploadFile) -> str:
    logging.info(f"進入 transcribe_audio 函式，處理檔案: {audio_file.filename}...")
    transcribed_text = ""
    audio_content = await audio_file.read()
    logging.info(f"接收到音訊數據，大小: {len(audio_content)} 字節。")
    
    start_time_overall = time.time()
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmpfile:
        tmpfile.write(audio_content)
        tmpfile_path = tmpfile.name
        logging.info(f"音訊臨時檔案已建立: {tmpfile_path}")

        try:
            logging.info("開始語音轉錄...")
            start_time_transcribe = time.time()
            result = whisper_model.transcribe(tmpfile_path, language="zh")
            end_time_transcribe = time.time()
            transcribed_text = result["text"]
            logging.info(f"語音轉錄成功。耗時: {end_time_transcribe - start_time_transcribe:.2f} 秒。轉錄內容: '{transcribed_text}'")
        except Exception as e:
            logging.error(f"Whisper 語音轉文字失敗: {e}。完整錯誤: {e}", exc_info=True)
            transcribed_text = ""
    end_time_overall = time.time()
    logging.info(f"transcribe_audio 函式執行完成，總耗時: {end_time_overall - start_time_overall:.2f} 秒。")
    return transcribed_text
