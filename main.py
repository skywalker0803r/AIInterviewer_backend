import os
import uuid
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import logging
import redis
import pickle
from speech_to_text import load_whisper_model
from interview_manager import InterviewManager
from job_scraper import get_jobs_from_104

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Global dictionary to hold interview manager instances, keyed by session_id (fallback if Redis is not used)
interview_sessions: Dict[str, InterviewManager] = {}

# Redis client
redis_client = None

@app.on_event("startup")
async def startup_event():
    global redis_client
    await load_whisper_model()
    redis_url = os.environ.get("REDIS")
    if not redis_url:
        logging.error("REDIS environment variable not set. Redis will not be used.")
    else:
        try:
            redis_client = redis.from_url(redis_url)
            redis_client.ping()
            logging.info("Connected to Redis.")
        except redis.exceptions.ConnectionError as e:
            logging.error(f"Could not connect to Redis: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    if redis_client:
        redis_client.close()
        logging.info("Disconnected from Redis.")

# --- CORS Configuration ---
# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aiinterviewer-frontend.onrender.com", "http://localhost", "http://127.0.0.1"], # Add localhost for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static Files ---
# Mount static files (e.g., generated audio files)
static_dir = "static"
os.makedirs(static_dir, exist_ok=True)
app.mount(f"/{static_dir}", StaticFiles(directory=static_dir), name="static")


# --- API Endpoints ---

@app.get("/jobs")
async def get_jobs(keyword: str = "前端工程師"):
    """
    Scrapes job listings from 104.com.tw based on a keyword.
    """
    try:
        jobs = await get_jobs_from_104(keyword)
        return {"jobs": jobs}
    except Exception as e:
        logging.error(f"Error getting jobs for keyword '{keyword}': {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve job listings.")

@app.post("/start_interview")
async def start_interview(request: Request):
    """
    Starts a new interview session.
    Creates a unique session_id and an InterviewManager instance.
    """
    global redis_client
    session_id = str(uuid.uuid4())
    
    try:
        body = await request.json()
        job_title = body.get("job", {}).get("title", "未知職缺")
        job_description = body.get("job_description", "")

        manager = InterviewManager()
        
        # Store manager in Redis
        if redis_client:
            redis_client.set(session_id, pickle.dumps(manager))
            logging.info(f"Stored session {session_id} in Redis.")
        else:
            # Fallback to in-memory if Redis is not available
            interview_sessions[session_id] = manager
            logging.warning(f"Redis not available, storing session {session_id} in memory.")
        
        # Start the preparation in the background (non-blocking)
        initial_response = await manager.start_new_interview(job_title, job_description, session_id)
        
        # Update manager in Redis after initial response
        if redis_client:
            redis_client.set(session_id, pickle.dumps(manager))

        logging.info(f"Started interview session {session_id} for job '{job_title}'")
        return JSONResponse({
            "message": "Interview session created. Ready to get the first question.",
            "session_id": session_id,
            "first_question": initial_response
        })

    except Exception as e:
        logging.error(f"Error starting interview: {e}", exc_info=True)
        if redis_client and redis_client.exists(session_id):
            redis_client.delete(session_id) # Clean up on failure
        elif session_id in interview_sessions:
            del interview_sessions[session_id] # Clean up on failure
        raise HTTPException(status_code=500, detail=f"Failed to start interview: {str(e)}")


@app.post("/submit_answer_and_get_next_question")
async def submit_answer_and_get_next_question(session_id: str = Form(...), audio_file: UploadFile = File(...), image_data: str = Form(...)):
    logging.info(f"Received request for /submit_answer_and_get_next_question for session {session_id}")
    manager = None
    if redis_client:
        manager_data = redis_client.get(session_id)
        if manager_data:
            manager = pickle.loads(manager_data)
    else:
        manager = interview_sessions.get(session_id)

    if not manager:
        raise HTTPException(status_code=404, detail="Interview session not found.")

    try:
        # 1. Process the user's spoken answer
        user_text = await manager.process_user_answer(session_id, audio_file, image_data)
        
        # 2. Get the next question from the AI
        next_question_data = await manager.get_next_question(session_id)
        next_question_data["user_text"] = user_text # Add user_text to the response
        
        # Update manager in Redis
        if redis_client:
            redis_client.set(session_id, pickle.dumps(manager))

        logging.info(f"Processed answer and got next question for session {session_id}")
        return JSONResponse(next_question_data)

    except Exception as e:
        logging.error(f"Error during interview loop for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/get_interview_report")
async def get_interview_report(session_id: str):
    """
    Retrieves the final interview report for a given session.
    """
    logging.info(f"Requesting report for session {session_id}")
    manager = None
    if redis_client:
        manager_data = redis_client.get(session_id)
        if manager_data:
            manager = pickle.loads(manager_data)
    else:
        manager = interview_sessions.get(session_id)

    if not manager:
        raise HTTPException(status_code=404, detail="Interview session not found.")
        
    report = manager.get_interview_report(session_id)
    if "error" in report:
        raise HTTPException(status_code=404, detail=report["error"])
        
    return JSONResponse(report)


@app.post("/end_interview")
async def end_interview(request: Request):
    """
    Explicitly ends an interview session and cleans up resources.
    """
    try:
        body = await request.json()
        session_id = body.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required.")
            
        if redis_client and redis_client.exists(session_id):
            redis_client.delete(session_id)
            logging.info(f"Successfully ended and cleaned up session {session_id} from Redis.")
            return JSONResponse({"message": f"Interview session {session_id} has been terminated."})
        elif session_id in interview_sessions:
            del interview_sessions[session_id]
            logging.info(f"Successfully ended and cleaned up session {session_id} from memory.")
            return JSONResponse({"message": f"Interview session {session_id} has been terminated."})
        else:
            raise HTTPException(status_code=404, detail="Interview session not found.")
            
    except Exception as e:
        logging.error(f"Error ending interview: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while ending the interview: {str(e)}")


