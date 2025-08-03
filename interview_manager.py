import json
import logging
import uuid
import time
import asyncio
from typing import Dict, Any

from config import GEMINI_API_KEY, EVALUATION_DIMENSIONS, GCS_BUCKET_NAME
from gemini_api import call_gemini_api, extract_json_from_gemini_response
from speech_to_text import transcribe_audio
from emotion_analysis import analyze_emotion
from text_to_speech import generate_and_upload_audio

interview_sessions: Dict[str, Dict[str, Any]] = {}

class InterviewManager:
    def __init__(self):
        pass

    async def start_new_interview(self, job_title: str, job_description: str) -> Dict[str, Any]:
        session_id = str(uuid.uuid4())
        interview_questions = await self._generate_interview_questions(job_title, job_description)

        interview_sessions[session_id] = {
            "job_title": job_title,
            "conversation_history": [],
            "interview_questions": interview_questions,
            "current_question_index": 0,
            "evaluation_dimensions": EVALUATION_DIMENSIONS,
            "evaluation_results": {dim: [] for dim in EVALUATION_DIMENSIONS},
            "audio_buffer": b"",
            "last_audio_time": time.time(),
            "latest_video_frame": None,
            "interview_completed": False
        }

        first_question = interview_questions[0]["question"]
        interview_sessions[session_id]["conversation_history"].append({"role": "model", "parts": [{"text": first_question}]})

        audio_url = await generate_and_upload_audio(first_question)

        return {
            "text": first_question,
            "audio_url": audio_url,
            "session_id": session_id,
            "total_questions": len(interview_questions)
        }

    async def _generate_interview_questions(self, job_title: str, job_description: str) -> list:
        question_generation_prompt = f"""你是一位專業的面試官。請根據應徵職位「{job_title}」以及以下職位描述：\n\n{job_description}\n\n設計 5 到 8 個面試問題。這些問題應該能夠評估候選人在以下方面的能力：{', '.join(EVALUATION_DIMENSIONS)}。請以 JSON 格式返回問題列表，每個問題包含 'id' 和 'question' 字段。例如：\n        {{"questions": [{{"id": 1, "question": "請自我介紹。"}}, {{"id": 2, "question": "..."}}]}}\n        """

        payload_questions = {
            "contents": [
                {"parts": [{"text": question_generation_prompt}]}
            ]
        }

        try:
            gemini_questions_reply = await call_gemini_api(GEMINI_API_KEY, payload_questions)
            logging.info(f"Gemini Questions API 回應：{json.dumps(gemini_questions_reply, ensure_ascii=False, indent=2)}")
            extracted_json_string = extract_json_from_gemini_response(gemini_questions_reply)
            questions_data = json.loads(extracted_json_string)
            interview_questions = questions_data.get("questions", [])
            logging.info(f"Generated interview questions count: {len(interview_questions)}")
            if not interview_questions:
                raise ValueError("Gemini generated an empty list of questions.")
            return interview_questions
        except (ValueError, json.JSONDecodeError) as e:
            logging.error(f"Failed to generate questions from Gemini or parse its response: {e}")
            # Fallback questions
            return [
                {"id": 1, "question": f"您好，歡迎您！我是今天的面試官。很高興您能來參加我們「{job_title}」職位的第一輪面試。首先，請您簡單介紹一下自己，並請您特別分享一下，您過去的經驗或專業背景，有哪些方面是您認為與我們『{job_title}』這個職位高度相關的？以及是什麼原因讓您對這個職位特別感興趣？"},
                {"id": 2, "question": "您認為自己最大的優點和缺點是什麼？這些特質如何影響您的工作？"},
                {"id": 3, "question": "在您過去的工作經驗中，有沒有遇到過什麼挑戰？您是如何克服這些挑戰的？"},
                {"id": 4, "question": "您對我們公司或這個職位有什麼了解？為什麼選擇我們公司？"},
                {"id": 5, "question": "您對未來的職業發展有什麼規劃？"}
            ]

    async def process_user_audio_and_video(self, session_id: str, websocket):
        session_data = interview_sessions.get(session_id)
        if not session_data:
            logging.error(f"Session ID {session_id} not found for audio processing.")
            return

        audio_buffer = session_data["audio_buffer"]
        if not audio_buffer:
            logging.debug(f"process_user_audio_and_video called for session {session_id} but buffer is empty.")
            return

        logging.info(f"Processing audio buffer for session {session_id}. Buffer size: {len(audio_buffer)} bytes.")

        session_data["audio_buffer"] = b""
        
        user_text = await transcribe_audio(audio_buffer)
        user_emotion = await analyze_emotion(session_data["latest_video_frame"])

        logging.info(f"Raw transcribed text from Whisper: '{user_text}'")

        if user_text.strip():
            logging.debug(f"User transcribed text: {user_text}")
            await websocket.send_json({"speaker": "你", "text": user_text})
            session_data["conversation_history"].append({"role": "user", "parts": [{"text": user_text}]})

            await self._evaluate_and_respond(session_id, user_text, user_emotion, websocket)

    async def _evaluate_and_respond(self, session_id: str, user_text: str, user_emotion: str, websocket):
        session_data = interview_sessions[session_id]
        current_question_index = session_data["current_question_index"]
        interview_questions = session_data["interview_questions"]
        evaluation_dimensions = session_data["evaluation_dimensions"]
        evaluation_results = session_data["evaluation_results"]
        job_title = session_data["job_title"]

        current_question_obj = interview_questions[current_question_index]
        current_question_text = current_question_obj["question"]

        evaluation_prompt = f"""你是一位專業的面試官，正在評估應徵「{job_title}」職位的候選人。候選人剛才回答了你的問題：「{current_question_text}」。
        候選人的回答是：「{user_text}」。
        根據面部情緒分析，候選人當前的情緒是：「{user_emotion}」。
        請你根據候選人的回答和面部情緒，對其在以下 {len(evaluation_dimensions)} 個維度的表現進行 1 到 5 分的評分（1分最低，5分最高）。
        評分維度：{', '.join(evaluation_dimensions)}
        請以 JSON 格式返回評分結果，例如：
        {{"技術深度": 4, "領導能力": 3, "溝通能力": 5, "抗壓能力": 4, "解決問題能力": 4, "學習能力": 4, "團隊合作": 5, "創新思維": 4}}
        以及你內心的想法
        """
        
        payload_evaluation = {
            "contents": [
                {"parts": [{"text": evaluation_prompt}]}
            ]
        }

        logging.debug("Calling Gemini API for evaluation...")
        gemini_evaluation_reply = await call_gemini_api(GEMINI_API_KEY, payload_evaluation)
        logging.debug(f"Gemini Evaluation API raw response: {json.dumps(gemini_evaluation_reply, ensure_ascii=False, indent=2)}")

        if "candidates" in gemini_evaluation_reply and gemini_evaluation_reply["candidates"]:
            evaluation_text = gemini_evaluation_reply["candidates"][0]["content"]["parts"][0]["text"]
            logging.info(f"Gemini 對回答的評語：\n{evaluation_text}")
            try:
                extracted_json_string = extract_json_from_gemini_response(gemini_evaluation_reply)
                scores = json.loads(extracted_json_string)
                for dim in evaluation_dimensions:
                    if dim in scores and isinstance(scores[dim], (int, float)):
                        evaluation_results[dim].append(scores[dim])
                logging.info(f"評分結果：{scores}")
            except (ValueError, json.JSONDecodeError) as e:
                logging.error(f"Gemini returned invalid JSON for evaluation: {evaluation_text}. Error: {e}")
        else:
            logging.warning(f"Gemini did not return evaluation results or candidates were empty. Raw response: {json.dumps(gemini_evaluation_reply, ensure_ascii=False, indent=2)}")

        session_data["current_question_index"] += 1
        if session_data["current_question_index"] < len(interview_questions):
            next_question_obj = interview_questions[session_data["current_question_index"]]
            gemini_response_text = next_question_obj["question"]
            session_data["conversation_history"].append({"role": "model", "parts": [{"text": gemini_response_text}]})
            audio_url = await generate_and_upload_audio(gemini_response_text)
            await websocket.send_json({"text": gemini_response_text, "audio_url": audio_url})
            logging.info("Sent AI response and audio URL to frontend.")
        else:
            final_message = "謝謝您今天來參加面試，面試到此結束。"
            session_data["conversation_history"].append({"role": "model", "parts": [{"text": final_message}]})
            audio_url = await generate_and_upload_audio(final_message)
            await websocket.send_json({"text": final_message, "audio_url": audio_url, "interview_ended": True})
            await asyncio.sleep(1) # Add a delay to ensure frontend receives the final message and audio
            session_data["interview_completed"] = True
            logging.info("Interview ended. Sent final message and audio URL to frontend.")
            await websocket.close() # Backend explicitly closes the WebSocket

    def get_interview_report(self, session_id: str) -> Dict[str, Any]:
        session_data = interview_sessions.get(session_id)
        if not session_data:
            logging.error(f"Session ID {session_id} not found for report generation.")
            return {"error": "Session not found"}

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
                dimension_scores[dim] = 0
                logging.debug(f"No scores for dimension {dim}.")

        if total_scores_count > 0:
            overall_score /= total_scores_count
        else:
            overall_score = 0

        hired = overall_score >= 3.5
        logging.info(f"Overall score for session {session_id}: {overall_score:.2f}, Hired: {hired}")

        # Clean up session after report is generated
        if session_id in interview_sessions:
            del interview_sessions[session_id]
            logging.debug(f"Session {session_id} cleaned up after report generation.")

        return {
            "overall_score": overall_score,
            "dimension_scores": dimension_scores,
            "hired": hired
        }

    def update_audio_buffer(self, session_id: str, audio_chunk: bytes):
        session_data = interview_sessions.get(session_id)
        if session_data:
            session_data["audio_buffer"] += audio_chunk
            session_data["last_audio_time"] = time.time()

    def update_video_frame(self, session_id: str, video_frame_base64: str):
        session_data = interview_sessions.get(session_id)
        if session_data:
            session_data["latest_video_frame"] = video_frame_base64

    def get_session_data(self, session_id: str):
        return interview_sessions.get(session_id)

    def remove_session(self, session_id: str):
        if session_id in interview_sessions:
            del interview_sessions[session_id]
            logging.info(f"Session {session_id} removed.")
