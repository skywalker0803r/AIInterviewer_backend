import logging
import json
from typing import Dict, Any, List
from fastapi import UploadFile

from config import GEMINI_API_KEY, EVALUATION_DIMENSIONS
from gemini_api import call_gemini_api, extract_json_from_gemini_response
from speech_to_text import transcribe_audio
from text_to_speech import generate_and_upload_audio
# from emotion_analysis import analyze_emotion # Temporarily disable for simplification

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
        """
        Initializes the interview by generating questions and preparing the first one.
        """
        self.job_title = job_title
        self.session_id = session_id
        logging.info(f"[{self.session_id}] Generating interview questions for '{self.job_title}'.")

        self.interview_questions = await self._generate_interview_questions(job_title, job_description)
        
        if not self.interview_questions:
            raise ValueError("Failed to generate or retrieve fallback interview questions.")

        # Prepare the first question
        first_question_text = self.interview_questions[0]["question"]
        self.conversation_history.append({"role": "model", "parts": [{"text": first_question_text}]})
        
        audio_url = await generate_and_upload_audio(first_question_text)
        
        logging.info(f"[{self.session_id}] First question is ready.")
        return {
            "text": first_question_text,
            "audio_url": audio_url,
            "total_questions": len(self.interview_questions)
        }

    async def process_user_answer(self, session_id: str, audio_file: UploadFile):
        """
        Processes the user's audio answer, transcribes it, and evaluates it.
        """
        if self.session_id != session_id:
            raise ValueError("Session ID mismatch.")

        logging.info(f"[{self.session_id}] Processing user answer.")
        user_text = await transcribe_audio(audio_file)
        logging.info(f"[{self.session_id}] Transcribed text: '{user_text}'")

        if not user_text.strip():
            logging.warning(f"[{self.session_id}] Transcription resulted in empty text.")
            # Optionally, handle empty transcription (e.g., ask user to repeat)
            return

        self.conversation_history.append({"role": "user", "parts": [{"text": user_text}]})
        await self._evaluate_answer(user_text)

    async def get_next_question(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieves the next question in the sequence.
        """
        if self.session_id != session_id:
            raise ValueError("Session ID mismatch.")

        self.current_question_index += 1
        if self.current_question_index < len(self.interview_questions):
            next_question_text = self.interview_questions[self.current_question_index]["question"]
            self.conversation_history.append({"role": "model", "parts": [{"text": next_question_text}]})
            
            audio_url = await generate_and_upload_audio(next_question_text)
            
            logging.info(f"[{self.session_id}] Prepared next question #{self.current_question_index + 1}.")
            return {"text": next_question_text, "audio_url": audio_url, "interview_ended": False}
        else:
            self.interview_completed = True
            final_message = "謝謝您今天來參加面試，面試到此結束。我們將在完成所有評估後通知您結果。"
            self.conversation_history.append({"role": "model", "parts": [{"text": final_message}]})
            
            audio_url = await generate_and_upload_audio(final_message)

            logging.info(f"[{self.session_id}] Interview has ended.")
            return {"text": final_message, "audio_url": audio_url, "interview_ended": True}

    def get_interview_report(self, session_id: str) -> Dict[str, Any]:
        """
        Generates and returns the final interview report.
        """
        if self.session_id != session_id or not self.interview_completed:
             return {"error": "Session not found or interview not completed."}

        logging.info(f"[{self.session_id}] Generating report.")
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

        return {
            "overall_score": overall_score,
            "dimension_scores": dimension_scores,
            "hired": hired,
            "conversation_history": self.conversation_history
        }

    async def _generate_interview_questions(self, job_title: str, job_description: str) -> List[Dict[str, Any]]:
        """Generates interview questions using the Gemini API."""
        prompt = f"""你是一位專業的AI面試官。請根據應徵職位「{job_title}」及職位描述「{job_description}」，設計5個核心面試問題。請以JSON格式返回，例如：{{"questions": [{{"id": 1, "question": "..."}}]}}."""
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = await call_gemini_api(GEMINI_API_KEY, payload)
            json_data = json.loads(extract_json_from_gemini_response(response))
            questions = json_data.get("questions", [])
            if not questions: raise ValueError("Empty question list")
            return questions
        except Exception as e:
            logging.error(f"Failed to generate questions, using fallback. Error: {e}")
            return [
                {"id": 1, "question": f"您好，請簡單自我介紹，並說明您為何對「{job_title}」這個職位感興趣。"},
                {"id": 2, "question": "根據您的理解，這個職位最重要的核心能力是什麼？請舉例說明您如何具備這些能力。"},
                {"id": 3, "question": "請分享一個您過去最成功的專案經驗，您在其中扮演什麼角色？"},
                {"id": 4, "question": "工作中遇到壓力或挑戰時，您通常如何應對？"},
                {"id": 5, "question": "對於未來的職涯，您有什麼樣的規劃？"}
            ]

    async def _evaluate_answer(self, user_text: str):
        """Evaluates the user's answer using the Gemini API."""
        question_text = self.interview_questions[self.current_question_index]["question"]
        prompt = f"""作為一個AI面試官，請根據問題「{question_text}」和候選人回答「{user_text}」，對以下維度進行1-5分評分: {', '.join(EVALUATION_DIMENSIONS)}。請以JSON格式返回，例如：{{"技術深度": 4, "溝通能力": 5}}."""
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = await call_gemini_api(GEMINI_API_KEY, payload)
            scores = json.loads(extract_json_from_gemini_response(response))
            for dim, score in scores.items():
                if dim in self.evaluation_results and isinstance(score, (int, float)):
                    self.evaluation_results[dim].append(score)
            logging.info(f"[{self.session_id}] Evaluated scores: {scores}")
        except Exception as e:
            logging.error(f"[{self.session_id}] Failed to evaluate answer. Error: {e}")

