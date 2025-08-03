import logging
import tempfile
import time
import base64
import os

async def analyze_emotion(video_frame_base64: str) -> str:
    emotion = "neutral"
    if not video_frame_base64 or ',' not in video_frame_base64:
        logging.warning("No valid video frame data or missing comma in base64 string. Skipping emotion analysis.")
        return emotion

    import tensorflow as tf
    from deepface import DeepFace

    img_tmpfile = None
    try:
        header, encoded = video_frame_base64.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        logging.info(f"Received candidate face screenshot, size: {len(image_bytes)} bytes")

        img_tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        img_tmpfile.write(image_bytes)
        img_tmpfile_path = img_tmpfile.name
        img_tmpfile.close()

        with tf.device('/CPU:0'):
            logging.info("Starting face recognition (CPU mode)...")
            start_time_deepface = time.time()
            demographies = DeepFace.analyze(img_tmpfile_path, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            end_time_deepface = time.time()
            logging.info(f"Face recognition completed. Time taken: {end_time_deepface - start_time_deepface:.2f} seconds.")
            if demographies and len(demographies) > 0:
                emotion = demographies[0]['dominant_emotion']
                logging.info(f"Deepface emotion recognition result (CPU): {emotion}")
    except Exception as e:
        logging.error(f"Deepface emotion recognition failed: {e}")
        emotion = "unknown"
    finally:
        if img_tmpfile and os.path.exists(img_tmpfile_path):
            os.unlink(img_tmpfile_path)
    return emotion
