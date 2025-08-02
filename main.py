import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import json
import asyncio

from config import BACKEND_PUBLIC_URL
from speech_to_text import load_whisper_model
from interview_manager import InterviewManager
from job_scraper import get_jobs_from_104

logging.basicConfig(level=logging.INFO)

app = FastAPI()
interview_manager = InterviewManager()

# Mount static files (e.g., audio files)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aiinterviewer-frontend.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logging.info("Application startup...")
    # Configure TensorFlow to allow memory growth on GPU
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        logging.info("Configured TensorFlow to allow memory growth on GPU.")
    else:
        logging.info("No GPU devices found for TensorFlow. DeepFace will run on CPU if used.")
    await load_whisper_model()
    logging.info("Application startup complete.")

@app.get("/jobs")
async def get_jobs(keyword: str = "前端工程師"):
    jobs = await get_jobs_from_104(keyword)
    return {"jobs": jobs}

@app.post("/start_interview")
async def start_interview(request: Request):
    try:
        body = await request.json()
        job = body.get("job", {})
        job_title = job.get("title", "未知職缺")
        job_description = body.get("job_description", "")

        response_data = await interview_manager.start_new_interview(job_title, job_description)
        return JSONResponse(response_data)

    except Exception as e:
        logging.exception("Error processing /start_interview")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logging.debug(f"WebSocket connection established for session {session_id}")
    
    if not interview_manager.get_session_data(session_id):
        logging.error(f"Session ID {session_id} not found. Closing WebSocket.")
        await websocket.close(code=1008, reason="Invalid session ID")
        return

    try:
        while True:
            message = await websocket.receive()
            
            if "bytes" in message:
                data = message["bytes"]
                interview_manager.update_audio_buffer(session_id, data)
            elif "text" in message:
                try:
                    json_message = json.loads(message["text"])
                    if json_message.get("type") == "end_interview":
                        logging.info(f"Received end_interview signal for session {session_id}. Closing WebSocket.")
                        await interview_manager.process_user_audio_and_video(session_id, websocket) # Process any remaining audio
                        await websocket.close()
                        return
                    elif json_message.get("type") == "video_frame":
                        interview_manager.update_video_frame(session_id, json_message.get("data"))
                    elif json_message.get("type") == "end_of_speech":
                        logging.info(f"Received end_of_speech signal for session {session_id}. Processing audio buffer.")
                        await interview_manager.process_user_audio_and_video(session_id, websocket)
                except json.JSONDecodeError as e:
                    logging.warning(f"Received non-JSON text message or invalid JSON: {message['text']}. Error: {e}")

    except WebSocketDisconnect:
        logging.info(f"Client disconnected from WebSocket for session {session_id}")
        session_data = interview_manager.get_session_data(session_id)
        if session_data and not session_data.get("interview_completed", False):
            interview_manager.remove_session(session_id)
            logging.info(f"Session {session_id} cleaned up on disconnect (interview not completed).")
        else:
            logging.info(f"Session {session_id} not cleaned up on disconnect as interview was completed.")
    except Exception as e:
        logging.error(f"WebSocket error for session {session_id}: {e}", exc_info=True)
        session_data = interview_manager.get_session_data(session_id)
        if session_data and not session_data.get("interview_completed", False):
            interview_manager.remove_session(session_id)
            logging.info(f"Session {session_id} cleaned up on error (interview not completed).")
        else:
            logging.info(f"Session {session_id} not cleaned up on error as interview was completed.")

@app.get("/get_interview_report")
async def get_interview_report(session_id: str):
    logging.info(f"Received request for interview report for session {session_id}")
    report = interview_manager.get_interview_report(session_id)
    if "error" in report:
        return JSONResponse(report, status_code=404)
    return JSONResponse(report)
