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
import time
import json

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# --- CORS Configuration ---
# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aiinterviewer-frontend.onrender.com", "http://localhost", "http://127.0.0.1"], # Add localhost for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dictionary to hold interview manager instances, keyed by session_id (fallback if Redis is not used)
interview_sessions: Dict[str, InterviewManager] = {}

# Redis client
redis_client = None

# --- Static Files ---
# Mount static files (e.g., generated audio files)
static_dir = "static"
os.makedirs(static_dir, exist_ok=True)
app.mount(f"/{static_dir}", StaticFiles(directory=static_dir), name="static")

# --- API Endpoints ---
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

@app.get("/jobs")
async def get_jobs(keyword: str = "前端工程師"):
    logging.info(f"收到職缺搜尋請求，關鍵字: '{keyword}'")
    start_time = time.time()
    try:
        jobs = await get_jobs_from_104(keyword)
        end_time = time.time()
        logging.info(f"職缺搜尋完成，找到 {len(jobs)} 個職缺，耗時: {end_time - start_time:.2f} 秒。")
        return {"jobs": jobs}
    except Exception as e:
        logging.error(f"搜尋職缺 '{keyword}' 時發生錯誤: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="無法檢索職缺列表。")

@app.post("/start_interview")
async def start_interview(request: Request):
    logging.info("收到啟動面試請求。")
    start_time = time.time()
    global redis_client
    session_id = str(uuid.uuid4())
    
    try:
        body = await request.json()
        job_title = body.get("job", {}).get("title", "未知職缺")
        job_description = body.get("job_description", "")
        logging.info(f"為新會話 {session_id} 準備面試，職位: '{job_title}'")

        manager = InterviewManager()
        
        # Start the preparation in the background (non-blocking)
        initial_response = await manager.start_new_interview(job_title, job_description, session_id)
        
        # Store manager state in Redis as JSON
        if redis_client:
            redis_client.set(session_id, json.dumps(manager.to_dict()))
            logging.info(f"會話 {session_id} 已儲存到 Redis。")
        else:
            # Fallback to in-memory if Redis is not available
            interview_sessions[session_id] = manager
            logging.warning(f"Redis 不可用，會話 {session_id} 儲存到記憶體中。")

        end_time = time.time()
        logging.info(f"面試會話 {session_id} 已啟動，耗時: {end_time - start_time:.2f} 秒。")
        return JSONResponse({
            "message": "面試會話已建立。準備好獲取第一個問題。",
            "session_id": session_id,
            "first_question": initial_response
        })

    except Exception as e:
        logging.error(f"啟動面試時發生錯誤: {e}", exc_info=True)
        if redis_client and redis_client.exists(session_id):
            redis_client.delete(session_id) # Clean up on failure
        elif session_id in interview_sessions:
            del interview_sessions[session_id] # Clean up on failure
        raise HTTPException(status_code=500, detail=f"無法啟動面試: {str(e)}")


@app.post("/submit_answer_and_get_next_question")
async def submit_answer_and_get_next_question(session_id: str = Form(...), audio_file: UploadFile = File(...), image_data: str = Form(...)):
    logging.info(f"收到會話 {session_id} 的答案提交請求。音訊檔案大小: {audio_file.size} 字節，圖像數據存在: {bool(image_data)}。")
    start_time = time.time()
    manager = None
    if redis_client:
        manager_data_json = redis_client.get(session_id)
        if manager_data_json:
            manager_data = json.loads(manager_data_json)
            manager = await InterviewManager.from_dict(manager_data)
    else:
        manager = interview_sessions.get(session_id)

    if not manager:
        logging.error(f"會話 {session_id} 未找到。")
        raise HTTPException(status_code=404, detail="面試會話未找到。")

    try:
        # 1. Process the user's spoken answer
        user_text = await manager.process_user_answer(session_id, audio_file, image_data)
        
        # 2. Get the next question from the AI
        next_question_data = await manager.get_next_question(session_id)
        next_question_data["user_text"] = user_text # Add user_text to the response
        
        # Update manager state in Redis
        if redis_client:
            redis_client.set(session_id, json.dumps(manager.to_dict()))

        end_time = time.time()
        logging.info(f"會話 {session_id} 的答案處理和下一個問題獲取完成，耗時: {end_time - start_time:.2f} 秒。")
        return JSONResponse(next_question_data)

    except Exception as e:
        logging.error(f"會話 {session_id} 的面試循環中發生錯誤: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"發生錯誤: {str(e)}")


@app.get("/get_interview_report")
async def get_interview_report(session_id: str):
    logging.info(f"收到獲取會話 {session_id} 報告的請求。")
    start_time = time.time()
    manager = None
    if redis_client:
        manager_data_json = redis_client.get(session_id)
        if manager_data_json:
            manager_data = json.loads(manager_data_json)
            manager = await InterviewManager.from_dict(manager_data)
    else:
        manager = interview_sessions.get(session_id)

    if not manager:
        logging.error(f"會話 {session_id} 未找到，無法生成報告。")
        raise HTTPException(status_code=404, detail="面試會話未找到。")
        
    report = manager.get_interview_report(session_id)
    if "error" in report:
        logging.error(f"會話 {session_id} 報告生成失敗: {report['error']}")
        raise HTTPException(status_code=404, detail=report["error"])
        
    end_time = time.time()
    logging.info(f"會話 {session_id} 報告已生成，耗時: {end_time - start_time:.2f} 秒。")
    return JSONResponse(report)


@app.post("/end_interview")
async def end_interview(request: Request):
    logging.info("收到結束面試請求。")
    start_time = time.time()
    try:
        body = await request.json()
        session_id = body.get("session_id")
        if not session_id:
            logging.error("結束面試請求缺少 session_id。")
            raise HTTPException(status_code=400, detail="session_id 是必需的。")
            
        if redis_client and redis_client.exists(session_id):
            redis_client.delete(session_id)
            end_time = time.time()
            logging.info(f"會話 {session_id} 已成功從 Redis 結束並清理，耗時: {end_time - start_time:.2f} 秒。")
            return JSONResponse({"message": f"面試會話 {session_id} 已終止。"})
        elif session_id in interview_sessions:
            del interview_sessions[session_id]
            end_time = time.time()
            logging.info(f"會話 {session_id} 已成功從記憶體中結束並清理，耗時: {end_time - start_time:.2f} 秒。")
            return JSONResponse({"message": f"面試會話 {session_id} 已終止。"})
        else:
            logging.error(f"嘗試結束不存在的會話 {session_id}。")
            raise HTTPException(status_code=404, detail="面試會話未找到。")
            
    except Exception as e:
        logging.error(f"結束面試時發生錯誤: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"結束面試時發生錯誤: {str(e)}")


