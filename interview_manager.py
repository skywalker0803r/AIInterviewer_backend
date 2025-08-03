import logging
import json
from typing import Dict, Any, List
from fastapi import UploadFile
import base64
import time
from config import GEMINI_API_KEY, EVALUATION_DIMENSIONS
from gemini_api import call_gemini_api, extract_json_from_gemini_response
from speech_to_text import transcribe_audio
from text_to_speech import generate_and_upload_audio
from emotion_analysis import analyze_emotion

class InterviewManager:
    """
    Manages the state and logic for a single interview session.
    This class is now stateless regarding session management; session persistence
    is handled by the caller (e.g., in main.py).
    """
    def __init__(self):
        # Instance variables are initialized here
        self.job_title: str = ""
        self.session_id: str = ""
        self.conversation_history: List[Dict[str, Any]] = []
        self.interview_questions: List[Dict[str, Any]] = []
        self.current_question_index: int = 0
        self.evaluation_results: Dict[str, List[float]] = {dim: [] for dim in EVALUATION_DIMENSIONS}
        self.interview_completed: bool = False

    async def start_new_interview(self, job_title: str, job_description: str, session_id: str) -> Dict[str, Any]:
        self.job_title = job_title
        self.session_id = session_id
        logging.info(f"[{self.session_id}] 開始為職位 '{self.job_title}' 生成面試問題。")
        start_time = time.time()

        self.interview_questions = await self._generate_interview_questions(job_title, job_description)
        
        if not self.interview_questions:
            logging.error(f"[{self.session_id}] 無法生成或檢索備用面試問題。")
            raise ValueError("無法生成或檢索備用面試問題。")

        first_question_text = self.interview_questions[0]["question"]
        self.conversation_history.append({"role": "model", "parts": [{"text": first_question_text}]})
        
        audio_url = await generate_and_upload_audio(first_question_text)
        
        end_time = time.time()
        logging.info(f"[{self.session_id}] 第一個問題已準備就緒，耗時: {end_time - start_time:.2f} 秒。")
        return {
            "text": first_question_text,
            "audio_url": audio_url,
            "total_questions": len(self.interview_questions)
        }

    async def process_user_answer(self, session_id: str, audio_file: UploadFile, image_data: str):
        if self.session_id != session_id:
            logging.error(f"[{self.session_id}] 會話ID不匹配。預期: {self.session_id}, 收到: {session_id}")
            raise ValueError("會話ID不匹配。")

        logging.info(f"[{self.session_id}] 開始處理使用者答案。")
        start_time_process = time.time()

        user_text = await transcribe_audio(audio_file)
        logging.info(f"[{self.session_id}] 語音轉錄結果: '{user_text}'")

        emotion_result = None
        if image_data:
            logging.info(f"[{self.session_id}] 檢測到圖像數據，開始情緒分析。")
            start_time_emotion = time.time()
            try:
                emotion_result = await analyze_emotion(image_data)
                end_time_emotion = time.time()
                logging.info(f"[{self.session_id}] 情緒分析結果: {emotion_result}，耗時: {end_time_emotion - start_time_emotion:.2f} 秒。")
            except Exception as e:
                logging.error(f"[{self.session_id}] 情緒分析失敗: {e}", exc_info=True)

        if not user_text.strip():
            logging.warning(f"[{self.session_id}] 轉錄內容為空。")
            # Optionally, handle empty transcription (e.g., ask user to repeat)
            # return

        self.conversation_history.append({"role": "user", "parts": [{"text": user_text}]})
        
        start_time_evaluate = time.time()
        await self._evaluate_answer(user_text, emotion_result)
        end_time_evaluate = time.time()
        logging.info(f"[{self.session_id}] 答案評估完成，耗時: {end_time_evaluate - start_time_evaluate:.2f} 秒。")

        end_time_process = time.time()
        logging.info(f"[{self.session_id}] 使用者答案處理總耗時: {end_time_process - start_time_process:.2f} 秒。")
        return user_text

    async def get_next_question(self, session_id: str) -> Dict[str, Any]:
        if self.session_id != session_id:
            logging.error(f"[{self.session_id}] 會話ID不匹配。預期: {self.session_id}, 收到: {session_id}")
            raise ValueError("會話ID不匹配。")

        logging.info(f"[{self.session_id}] 準備獲取下一個問題。當前問題索引: {self.current_question_index}。")
        start_time = time.time()

        self.current_question_index += 1
        if self.current_question_index < len(self.interview_questions):
            next_question_text = self.interview_questions[self.current_question_index]["question"]
            self.conversation_history.append({"role": "model", "parts": [{"text": next_question_text}]})
            
            audio_url = await generate_and_upload_audio(next_question_text)
            
            end_time = time.time()
            logging.info(f"[{self.session_id}] 已準備好下一個問題 #{self.current_question_index + 1}，耗時: {end_time - start_time:.2f} 秒。")
            return {"text": next_question_text, "audio_url": audio_url, "interview_ended": False}
        else:
            self.interview_completed = True
            final_message = "謝謝您今天來參加面試，面試到此結束。我們將在完成所有評估後通知您結果。"
            self.conversation_history.append({"role": "model", "parts": [{"text": final_message}]})
            
            audio_url = await generate_and_upload_audio(final_message)

            end_time = time.time()
            logging.info(f"[{self.session_id}] 面試已結束，耗時: {end_time - start_time:.2f} 秒。")
            return {"text": final_message, "audio_url": audio_url, "interview_ended": True}

    def get_interview_report(self, session_id: str) -> Dict[str, Any]:
        if self.session_id != session_id or not self.interview_completed:
            logging.error(f"[{self.session_id}] 無法生成報告：會話ID不匹配或面試未完成。")
            return {"error": "會話未找到或面試未完成。"}

        logging.info(f"[{self.session_id}] 開始生成面試報告。")
        start_time = time.time()
        dimension_scores = {}
        total_scores_count = 0
        overall_score = 0.0

        for dim in EVALUATION_DIMENSIONS:
            scores = self.evaluation_results.get(dim, [])
            if scores:
                avg_score = sum(scores) / len(scores)
                dimension_scores[dim] = avg_score
                overall_score += avg_score
                total_scores_count += 1
            else:
                dimension_scores[dim] = 0

        if total_scores_count > 0:
            overall_score /= total_scores_count

        hired = overall_score >= 3.5

        end_time = time.time()
        logging.info(f"[{self.session_id}] 面試報告生成完成，耗時: {end_time - start_time:.2f} 秒。")
        return {
            "overall_score": overall_score,
            "dimension_scores": dimension_scores,
            "hired": hired,
            "conversation_history": self.conversation_history
        }

    async def _generate_interview_questions(self, job_title: str, job_description: str) -> List[Dict[str, Any]]:
        logging.info(f"[{self.session_id}] 開始生成面試問題，職位: '{job_title}'。")
        start_time = time.time()
        prompt = f"""你是一位專業的AI面試官。請根據應徵職位「{job_title}」及職位描述「{job_description}」，設計2個核心面試問題。請以JSON格式返回，例如：{{"questions": [{{"id": 1, "question": "..."}}]}}."""
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = await call_gemini_api(GEMINI_API_KEY, payload)
            json_data = json.loads(extract_json_from_gemini_response(response))
            questions = json_data.get("questions", [])
            if not questions: 
                logging.warning(f"[{self.session_id}] Gemini API 返回空問題列表。")
                raise ValueError("空問題列表")
            end_time = time.time()
            logging.info(f"[{self.session_id}] 面試問題生成完成，共 {len(questions)} 個問題，耗時: {end_time - start_time:.2f} 秒。")
            return questions
        except Exception as e:
            logging.error(f"[{self.session_id}] 生成問題失敗，使用備用問題。錯誤: {e}", exc_info=True)
            return [
                {"id": 1, "question": f"您好，請簡單自我介紹，並說明您為何對「{job_title}」這個職位感興趣。"},
                {"id": 2, "question": "根據您的理解，這個職位最重要的核心能力是什麼？請舉例說明您如何具備這些能力。"},
                {"id": 3, "question": "請分享一個您過去最成功的專案經驗，您在其中扮演什麼角色？"},
                {"id": 4, "question": "工作中遇到壓力或挑戰時，您通常如何應對？"},
                {"id": 5, "question": "對於未來的職涯，您有什麼樣的規劃？"}
            ]

    async def _evaluate_answer(self, user_text: str, emotion_result: Dict[str, Any] = None):
        logging.info(f"[{self.session_id}] 開始評估使用者答案。")
        start_time = time.time()
        question_text = self.interview_questions[self.current_question_index]["question"]
        
        emotion_info = ""
        if emotion_result:
            emotion_info = f"\n候選人臉部情緒分析結果：{emotion_result}"

        prompt = f"""作為一個AI面試官，請根據問題「{question_text}」、候選人回答「{user_text}」{emotion_info}，對以下維度進行1-5分評分: {', '.join(EVALUATION_DIMENSIONS)}。同時，請提供對候選人回答的詳細分析和理由，並綜合考慮其情緒表現。請以JSON格式返回，例如：{{"scores": {{"技術深度": 4, "溝通能力": 5}}, "reasoning": "候選人在技術深度方面表現良好，因為..."}}."""
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = await call_gemini_api(GEMINI_API_KEY, payload)
            json_data = json.loads(extract_json_from_gemini_response(response))
            scores = json_data.get("scores", {})
            reasoning = json_data.get("reasoning", "")

            for dim, score in scores.items():
                if dim in self.evaluation_results:
                    self.evaluation_results[dim].append(score)
                else:
                    logging.warning(f"[{self.session_id}] 收到未知評估維度: {dim}")
            
            logging.info(f"[{self.session_id}] Gemini 評估結果 - 分數: {scores}")
            if reasoning:
                logging.info(f"[{self.session_id}] Gemini 評估結果 - 分析: {reasoning}")
                self.conversation_history.append({"role": "model", "parts": [{"text": f"AI 評估: {reasoning}"}]})

        except Exception as e:
            logging.error(f"[{self.session_id}] 評估答案時呼叫 Gemini API 失敗: {e}", exc_info=True)
            # Fallback: If evaluation fails, add a neutral score or skip
            for dim in EVALUATION_DIMENSIONS:
                self.evaluation_results[dim].append(3) # Neutral score if evaluation fails

