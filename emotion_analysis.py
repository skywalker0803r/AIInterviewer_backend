import logging
import tempfile
import time
import base64
import os

async def analyze_emotion(video_frame_base64: str) -> str:
    emotion = "neutral"
    logging.info(f"進入 analyze_emotion 函式。接收到圖像數據長度: {len(video_frame_base64)}。")
    if not video_frame_base64 or ',' not in video_frame_base64:
        logging.warning("無效的視訊幀數據或 Base64 字串中缺少逗號。跳過情緒分析。")
        return emotion

    import tensorflow as tf
    from deepface import DeepFace

    img_tmpfile = None
    try:
        header, encoded = video_frame_base64.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        logging.info(f"已解碼 Base64 圖像數據，大小: {len(image_bytes)} 字節。")

        img_tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        img_tmpfile.write(image_bytes)
        img_tmpfile_path = img_tmpfile.name
        img_tmpfile.close()
        logging.info(f"圖像臨時檔案已建立: {img_tmpfile_path}")

        with tf.device('/CPU:0'):
            logging.info("開始執行 DeepFace 情緒分析 (CPU 模式)...")
            start_time_deepface = time.time()
            demographies = DeepFace.analyze(img_tmpfile_path, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            end_time_deepface = time.time()
            logging.info(f"DeepFace 情緒分析完成。耗時: {end_time_deepface - start_time_deepface:.2f} 秒。")
            if demographies and len(demographies) > 0:
                emotion = demographies[0]['dominant_emotion']
                logging.info(f"DeepFace 情緒分析結果: {emotion}")
    except Exception as e:
        logging.error(f"DeepFace 情緒分析失敗: {e}", exc_info=True)
        emotion = "unknown"
    finally:
        if img_tmpfile and os.path.exists(img_tmpfile_path):
            os.unlink(img_tmpfile_path)
            logging.info(f"已刪除圖像臨時檔案: {img_tmpfile_path}")
    logging.info(f"analyze_emotion 函式執行完成，返回情緒: {emotion}。")
    return emotion
