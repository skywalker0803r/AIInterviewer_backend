from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
from bs4 import BeautifulSoup
from fastapi.staticfiles import StaticFiles
import logging
import json
import re
import os
from gtts import gTTS
import uuid
from dotenv import load_dotenv
import whisper
import tempfile
import time # Import time for performance logging
from deepface import DeepFace # 雖然目前沒有影像串流，但先引入
#123
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # 請設在 .env 或環境變數

logging.basicConfig(level=logging.INFO) # Change to INFO for less verbose logging

app = FastAPI()
# 提供靜態檔案（TTS 音訊）
app.mount("/static", StaticFiles(directory="static"), name="static")

# 允許前端跨域請求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或具體設為 ["http://127.0.0.1:5500"]（如果你用 VSCode Live Server）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

whisper_model = None

interview_sessions = {} # New: To store conversation history and job info per session

@app.on_event("startup")
async def load_whisper_model():
    global whisper_model
    logging.info("Loading Whisper model...")
    # Check for GPU availability
    import torch
    if torch.cuda.is_available():
        logging.info("CUDA is available. Using GPU for Whisper.")
        whisper_model = whisper.load_model("turbo", device="cuda")
    else:
        logging.info("CUDA is not available. Using CPU for Whisper.")
        whisper_model = whisper.load_model("turbo", device="cpu")
    logging.info("Whisper model loaded.")


@app.get("/jobs")
async def get_jobs(keyword: str = "前端工程師"):
    logging.info(f"接收到搜尋職缺關鍵字：{keyword}")
    
    url = (
        "https://www.104.com.tw/jobs/search/list"
        f"?ro=0&keyword={keyword}&jobcatExpMore=1&order=11&page=1"
    )
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.104.com.tw/jobs/search/"
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers)
        logging.info(f"104 JSON API 回應狀態碼：{resp.status_code}")
        if resp.status_code != 200:
            return {"jobs": []}

        try:
            data = resp.json()
            job_list = data.get("data", {}).get("list", [])
            logging.info(f"擷取到職缺數量：{len(job_list)}")

            result = []
            for job in job_list[:10]:
                logging.info(f"Full job object from 104 API: {job}") # Add this line
                job_no = job.get('jobNo')
                relative_job_url = job.get('link', {}).get('job')
                if relative_job_url:
                    job_url = f"https:{relative_job_url}"
                else:
                    job_url = f"https://www.104.com.tw/job/{job_no}" # Fallback, though it's likely incorrect
                logging.info(f"Processing job: {job.get('jobName')}, jobNo: {job_no}, Corrected URL: {job_url}")
                result.append({
                    "title": job.get("jobName"),
                    "company": job.get("custName"),
                    "url": job_url,
                    "description": job.get("description", "") # Add description
                })
            return {"jobs": result}
        except Exception as e:
            logging.error(f"解析 JSON 發生錯誤：{e}")
            return {"jobs": []}
        

async def call_gemini_api(client,payload):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    timeout = httpx.Timeout(30.0, read=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()
        except httpx.ReadTimeout:
            logging.error("連線到 Gemini API 時讀取超時")
            return None
        except Exception as e:
            logging.error(f"呼叫 Gemini API 發生錯誤: {e}")
            return None

@app.post("/start_interview")
async def start_interview(request: Request):
    try:
        body = await request.json()
        job = body.get("job", {})
        job_title = job.get("title", "未知職缺")
        job_description = body.get("job_description", "") # Get job description from request body

        session_id = str(uuid.uuid4()) # Generate a unique session ID
        
        # Define evaluation dimensions
        evaluation_dimensions = [
            "技術深度", "領導能力", "溝通能力", "抗壓能力",
            "解決問題能力", "學習能力", "團隊合作", "創新思維"
        ]

        # Prompt to generate interview questions based on job title, job description and evaluation dimensions
        question_generation_prompt = f"""你是一位專業的面試官。請根據應徵職位「{job_title}」以及以下職位描述：\n\n{job_description}\n\n設計 5 到 8 個面試問題。這些問題應該能夠評估候選人在以下方面的能力：{', '.join(evaluation_dimensions)}。請以 JSON 格式返回問題列表，每個問題包含 'id' 和 'question' 字段。例如：\n        {{"questions": [{{"id": 1, "question": "請自我介紹。"}}, {{"id": 2, "question": "..."}}]}}\n        """

        payload_questions = {
            "contents": [
                {"parts": [{"text": question_generation_prompt}]}
            ]
        }

        async with httpx.AsyncClient() as client:
            gemini_questions_reply = await call_gemini_api(client, payload_questions)
            logging.info(f"Gemini Questions API 回應：{json.dumps(gemini_questions_reply, ensure_ascii=False, indent=2)}")

            if "candidates" not in gemini_questions_reply or not gemini_questions_reply["candidates"]:
                logging.error(f"Gemini Questions API did not return candidates or candidates list is empty: {json.dumps(gemini_questions_reply, ensure_ascii=False, indent=2)}")
                raise ValueError("API 回應中缺少 candidates 或 candidates 為空，請檢查 API key 和參數")

            questions_text = gemini_questions_reply["candidates"][0]["content"]["parts"][0]["text"]
            
            # Extract JSON string from markdown code block
            json_match = re.search(r"```json\n([\s\S]*?)\n```", questions_text)
            if json_match:
                extracted_json_string = json_match.group(1)
            else:
                extracted_json_string = questions_text # Fallback if no markdown block

            try:
                questions_data = json.loads(extracted_json_string)
                interview_questions = questions_data.get("questions", [])
            except json.JSONDecodeError:
                logging.error(f"Gemini 返回的問題不是有效的 JSON: {questions_text}")
                interview_questions = [] # Fallback to empty list

            logging.info(f"Generated interview questions count: {len(interview_questions)}")

            if not interview_questions:
                # Fallback if Gemini fails to generate questions or returns empty list
                interview_questions = [
                    {"id": 1, "question": f"您好，歡迎您！我是今天的面試官。很高興您能來參加我們「{job_title}」職位的第一輪面試。首先，請您簡單介紹一下自己，並請您特別分享一下，您過去的經驗或專業背景，有哪些方面是您認為與我們『{job_title}』這個職位高度相關的？以及是什麼原因讓您對這個職位特別感興趣？"},
                    {"id": 2, "question": "您認為自己最大的優點和缺點是什麼？這些特質如何影響您的工作？"},
                    {"id": 3, "question": "在您過去的工作經驗中，有沒有遇到過什麼挑戰？您是如何克服這些挑戰的？"},
                    {"id": 4, "question": "您對我們公司或這個職位有什麼了解？為什麼選擇我們公司？"},
                    {"id": 5, "question": "您對未來的職業發展有什麼規劃？"}
                ]

            # Initialize conversation history and interview state for this session
            interview_sessions[session_id] = {
                "job_title": job_title,
                "conversation_history": [],
                "interview_questions": interview_questions,
                "current_question_index": 0,
                "evaluation_dimensions": evaluation_dimensions,
                "evaluation_results": {dim: [] for dim in evaluation_dimensions}
            }

            # Get the first question
            first_question = interview_questions[0]["question"]

            # Add Gemini's first question to the conversation history
            interview_sessions[session_id]["conversation_history"].append({"role": "model", "parts": [{"text": first_question}]})

            logging.info(f"Gemini 回覆內容：{first_question}")
            tts = gTTS(first_question, lang="zh-TW")
            audio_filename = f"{uuid.uuid4().hex}.mp3"
            audio_path = f"static/audio/{audio_filename}"
            os.makedirs("static/audio", exist_ok=True)
            tts.save(audio_path)

            return JSONResponse({
                "text": first_question,
                "audio_url": f"http://127.0.0.1:8002/static/audio/{audio_filename}",
                "session_id": session_id # Return the session ID
            })

    except Exception as e:
        logging.exception("處理 /start_interview 發生錯誤")
        return JSONResponse({"error": str(e)}, status_code=500)

async def process_user_input(audio_chunk: bytes, video_frame: bytes = None):
    transcribed_text = ""
    emotion = "neutral"

    # 將音訊資料寫入臨時檔案
    with tempfile.NamedTemporaryFile(delete=True, suffix=".webm") as tmpfile:
        tmpfile.write(audio_chunk)
        tmpfile_path = tmpfile.name

        try:
            start_time = time.time()
            # 使用 Whisper 進行語音轉文字
            result = whisper_model.transcribe(tmpfile_path, language="zh")
            end_time = time.time()
            transcribed_text = result["text"]
            logging.info(f"Whisper 轉錄結果: {transcribed_text}")
            logging.info(f"Whisper transcription took {end_time - start_time:.2f} seconds.")
        except Exception as e:
            logging.error(f"Whisper 語音轉文字失敗: {e}")

    # TODO: Implement Deepface emotion recognition here if video_frame is provided
    # if video_frame:
    #     try:
    #         # 將影像資料寫入臨時檔案
    #         with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as img_tmpfile:
    #             img_tmpfile.write(video_frame)
    #             img_tmpfile_path = img_tmpfile.name
    #             demographies = DeepFace.analyze(img_tmpfile_path, actions=['emotion'], enforce_detection=False)
    #             if demographies and len(demographies) > 0:
    #                 emotion = demographies[0]['dominant_emotion']
    #                 logging.info(f"Deepface 情緒辨識結果: {emotion}")
    #     except Exception as e:
    #         logging.error(f"Deepface 情緒辨識失敗: {e}")

    return transcribed_text, emotion

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logging.debug(f"WebSocket connection established for session {session_id}")
    
    if session_id not in interview_sessions:
        logging.error(f"Session ID {session_id} not found. Closing WebSocket.")
        await websocket.close(code=1008, reason="Invalid session ID")
        return

    session_data = interview_sessions[session_id]
    conversation_history = session_data["conversation_history"]
    job_title = session_data["job_title"]
    interview_questions = session_data["interview_questions"]
    current_question_index = session_data["current_question_index"]
    evaluation_dimensions = session_data["evaluation_dimensions"]
    evaluation_results = session_data["evaluation_results"]

    try:
        while True:
            logging.debug("Waiting for data from frontend...")
            message = await websocket.receive()
            logging.debug(f"Received WebSocket message: {message.keys()}")
            
            if "text" in message: # Handle text messages (e.g., end_interview signal)
                try:
                    json_message = json.loads(message["text"])
                    if json_message.get("type") == "end_interview":
                        logging.info(f"Received end_interview signal for session {session_id}. Closing WebSocket.")
                        await websocket.close()
                        logging.info(f"WebSocket closed for session {session_id} after end_interview signal.")
                        return
                except json.JSONDecodeError as e:
                    logging.warning(f"Received non-JSON text message or invalid JSON: {message['text']}. Error: {e}")

            elif "bytes" in message: # Handle binary messages (audio data)
                data = message["bytes"]
                logging.info(f"Received audio chunk of size: {len(data)} bytes")

                user_text, user_emotion = await process_user_input(data)

                if user_text.strip(): # 只有當使用者有說話時才進行處理
                    logging.debug(f"User transcribed text: {user_text}")
                    # 將使用者轉錄的文字發送回前端顯示
                    await websocket.send_json({"speaker": "你", "text": user_text})

                    # 將使用者輸入加入對話歷史
                    conversation_history.append({"role": "user", "parts": [{"text": user_text}]})

                    # --- 評分邏輯 --- #
                    logging.debug("Starting evaluation logic...")
                    current_question_obj = interview_questions[current_question_index]
                    current_question_text = current_question_obj["question"]

                    evaluation_prompt = f"""你是一位專業的面試官，正在評估應徵「{job_title}」職位的候選人。候選人剛才回答了你的問題：「{current_question_text}」。
                    候選人的回答是：「{user_text}」。
                    請你根據候選人的回答，對其在以下 8 個維度的表現進行 1 到 5 分的評分（1分最低，5分最高）。
                    評分維度：{', '.join(evaluation_dimensions)}
                    請以 JSON 格式返回評分結果，例如：
                    {{"技術深度": 4, "領導能力": 3, "溝通能力": 5, "抗壓能力": 4, "解決問題能力": 4, "學習能力": 4, "團隊合作": 5, "創新思維": 4}}
                    """
                    
                    payload_evaluation = {
                        "contents": [
                            {"parts": [{"text": evaluation_prompt}]}
                        ]
                    }

                    logging.debug("Calling Gemini API for evaluation...")
                    async with httpx.AsyncClient() as client:
                        gemini_evaluation_reply = await call_gemini_api(client, payload_evaluation)
                        logging.debug(f"Gemini Evaluation API raw response: {json.dumps(gemini_evaluation_reply, ensure_ascii=False, indent=2)}")

                        if "candidates" in gemini_evaluation_reply and gemini_evaluation_reply["candidates"]:
                            evaluation_text = gemini_evaluation_reply["candidates"][0]["content"]["parts"][0]["text"]
                            # Extract JSON string from markdown code block
                            json_match = re.search(r"```json\n([\s\S]*?)\n```", evaluation_text)
                            if json_match:
                                extracted_json_string = json_match.group(1)
                                logging.info(f"Extracted JSON string for evaluation: {extracted_json_string}")
                            else:
                                extracted_json_string = evaluation_text # Fallback if no markdown block
                                logging.warning(f"No JSON markdown block found for evaluation. Using raw text: {extracted_json_string}")

                            try:
                                scores = json.loads(extracted_json_string)
                                for dim in evaluation_dimensions:
                                    if dim in scores and isinstance(scores[dim], (int, float)):
                                        evaluation_results[dim].append(scores[dim])
                                logging.info(f"評分結果：{scores}")
                            except json.JSONDecodeError as e:
                                logging.error(f"Gemini 返回的評分不是有效的 JSON: {evaluation_text}. Error: {e}")
                        else:
                            logging.warning(f"Gemini 未返回評分結果或 candidates 為空。原始回應: {json.dumps(gemini_evaluation_reply, ensure_ascii=False, indent=2)}")

                    # --- 面試流程控制 --- #
                    logging.debug(f"Current question index: {current_question_index}, Total questions: {len(interview_questions)}")
                    session_data["current_question_index"] += 1
                    if session_data["current_question_index"] < len(interview_questions):
                        logging.debug("Moving to next question.")
                        # Ask next question
                        next_question_obj = interview_questions[session_data["current_question_index"]]
                        gemini_response_text = next_question_obj["question"]
                        conversation_history.append({"role": "model", "parts": [{"text": gemini_response_text}]})

                        logging.info(f"Gemini 回覆內容：{gemini_response_text}")
                        tts = gTTS(gemini_response_text, lang="zh-TW")
                        audio_filename = f"{uuid.uuid4().hex}.mp3"
                        audio_path = f"static/audio/{audio_filename}"
                        os.makedirs("static/audio", exist_ok=True)
                        tts.save(audio_path)
                        audio_url = f"http://127.0.0.1:8002/static/audio/{audio_filename}"

                        await websocket.send_json({"text": gemini_response_text, "audio_url": audio_url})
                    else:
                        # End of interview
                        logging.info("All questions answered. Ending interview.")
                        final_message = "謝謝您今天來參加面試，面試到此結束。"
                        conversation_history.append({"role": "model", "parts": [{"text": final_message}]})

                        logging.info(f"面試結束訊息：{final_message}")
                        tts = gTTS(final_message, lang="zh-TW")
                        audio_filename = f"{uuid.uuid4().hex}.mp3"
                        audio_path = f"static/audio/{audio_filename}"
                        os.makedirs("static/audio", exist_ok=True)
                        tts.save(audio_path)
                        audio_url = f"http://127.0.0.1:8002/static/audio/{audio_filename}"

                        await websocket.send_json({"text": final_message, "audio_url": audio_url, "interview_ended": True})
                        logging.debug(f"Sent interview_ended signal for session {session_id}.")
                        # Optionally, send evaluation results here or trigger a separate endpoint call from frontend
                        # For now, frontend will call /get_interview_report

            else:
                logging.warning(f"Received unknown WebSocket message type: {message}")

    except WebSocketDisconnect:
        logging.info(f"Client disconnected from WebSocket for session {session_id}")
        if session_id in interview_sessions:
            del interview_sessions[session_id] # Clean up session on disconnect
            logging.info(f"Session {session_id} cleaned up on disconnect.")
    except Exception as e:
        logging.error(f"WebSocket error for session {session_id}: {e}", exc_info=True) # Log full traceback
        if session_id in interview_sessions:
            del interview_sessions[session_id] # Clean up session on error
            logging.info(f"Session {session_id} cleaned up on error.")

@app.get("/get_interview_report")
async def get_interview_report(session_id: str):
    logging.info(f"Received request for interview report for session {session_id}")
    if session_id not in interview_sessions:
        logging.error(f"Session ID {session_id} not found for report generation. It might have been cleaned up already.")
        return JSONResponse({"error": "Session not found"}, status_code=404)

    session_data = interview_sessions[session_id]
    evaluation_dimensions = session_data["evaluation_dimensions"]
    evaluation_results = session_data["evaluation_results"]

    dimension_scores = {}
    overall_score = 0
    total_scores_count = 0

    for dim in evaluation_dimensions:
        if evaluation_results[dim]:
            avg_score = sum(evaluation_results[dim]) / len(evaluation_results[dim])
            dimension_scores[dim] = avg_score
            overall_score += avg_score
            total_scores_count += 1
            logging.debug(f"Dimension {dim} scores: {evaluation_results[dim]}, Avg: {avg_score:.2f}")
        else:
            dimension_scores[dim] = 0 # No scores for this dimension
            logging.debug(f"No scores for dimension {dim}.")

    if total_scores_count > 0:
        overall_score /= total_scores_count
    else:
        overall_score = 0

    # Simple hiring decision based on overall score
    hired = overall_score >= 3.5 # Threshold for hiring
    logging.info(f"Overall score for session {session_id}: {overall_score:.2f}, Hired: {hired}")

    # Clean up session after report is generated
    if session_id in interview_sessions:
        del interview_sessions[session_id]
        logging.debug(f"Session {session_id} cleaned up after report generation.")

    return JSONResponse({
        "overall_score": overall_score,
        "dimension_scores": dimension_scores,
        "hired": hired
    })