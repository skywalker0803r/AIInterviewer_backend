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

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter


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
        self.evaluation_results: Dict[str, List[float]] = {dim: [] for dim in EVALUATION_DIMENSIONS}
        self.interview_completed: bool = False

        # LangChain components
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.7)
        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        self.conversation = ConversationChain(llm=self.llm, memory=self.memory, verbose=False)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        self.vectorstore = None # Will be initialized with job description
        self.retriever = None

    def to_dict(self):
        # Only return serializable attributes
        return {
            "job_title": self.job_title,
            "session_id": self.session_id,
            "conversation_history": self.conversation_history,
            "evaluation_results": self.evaluation_results,
            "interview_completed": self.interview_completed,
            "job_description": self._original_job_description # Store original job_description
        }

    @classmethod
    async def from_dict(cls, data: Dict[str, Any]):
        manager = cls() # Create a new instance
        manager.job_title = data.get("job_title", "")
        manager.session_id = data.get("session_id", "")
        manager.conversation_history = data.get("conversation_history", [])
        manager.evaluation_results = data.get("evaluation_results", {dim: [] for dim in EVALUATION_DIMENSIONS})
        manager.interview_completed = data.get("interview_completed", False)
        manager._original_job_description = data.get("job_description", "")

        # Re-initialize LangChain components
        if manager._original_job_description:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_text(manager._original_job_description)
            manager.vectorstore = FAISS.from_texts(texts, manager.embeddings)
            manager.retriever = manager.vectorstore.as_retriever()
            logging.info(f"[{manager.session_id}] 職位描述已從字典重新載入到向量儲存中。")

        # Re-populate LangChain memory from conversation_history
        manager.memory.clear()
        for msg in manager.conversation_history:
            if msg["role"] == "user":
                manager.memory.chat_memory.add_user_message(msg["parts"][0]["text"])
            elif msg["role"] == "model":
                manager.memory.chat_memory.add_ai_message(msg["parts"][0]["text"])
        
        return manager

    async def start_new_interview(self, job_title: str, job_description: str, session_id: str) -> Dict[str, Any]:
        self.job_title = job_title
        self.session_id = session_id
        self._original_job_description = job_description # Store original job_description
        logging.info(f"[{self.session_id}] 開始為職位 '{self.job_title}' 生成面試問題。")
        start_time = time.time()

        # Initialize vector store with job description for RAG
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(job_description)
        self.vectorstore = FAISS.from_texts(texts, self.embeddings)
        self.retriever = self.vectorstore.as_retriever()
        logging.info(f"[{self.session_id}] 職位描述已載入到向量儲存中。")

        # Dynamically generate the first question
        first_question_text = await self._generate_dynamic_question(job_title, job_description, is_first_question=True)
        
        if not first_question_text:
            logging.error(f"[{self.session_id}] 無法生成第一個面試問題。")
            raise ValueError("無法生成第一個面試問題。")

        self.conversation_history.append({"role": "model", "parts": [{"text": first_question_text}]})
        
        # Add first question to LangChain memory
        self.memory.chat_memory.add_ai_message(first_question_text)

        audio_url = await generate_and_upload_audio(first_question_text)
        
        end_time = time.time()
        logging.info(f"[{self.session_id}] 第一個問題已準備就緒，耗時: {end_time - start_time:.2f} 秒。")
        return {
            "text": first_question_text,
            "audio_url": audio_url,
            # Removed total_questions as it's now dynamic
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
        # Add user's answer to LangChain memory
        self.memory.chat_memory.add_user_message(user_text)
        
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

        logging.info(f"[{self.session_id}] 準備獲取下一個問題。")
        start_time = time.time()

        # Use LangChain ConversationChain to generate the next question
        # We need to provide a prompt that guides the AI to ask the next question
        # or indicate interview completion.
        
        # Retrieve relevant context from job description for the next question
        retrieved_docs = []
        if self.retriever:
            # Use the last user message as query for retrieval
            last_user_message = self.memory.chat_memory.messages[-1].content if self.memory.chat_memory.messages else ""
            retrieved_docs = self.retriever.get_relevant_documents(last_user_message)
            logging.info(f"[{self.session_id}] RAG 檢索到 {len(retrieved_docs)} 份相關文件用於生成下一個問題。")
        
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Define a custom prompt template for the conversation chain
        # This prompt guides the AI to act as an interviewer and decide when to end the interview.
        template = f"""你是一位專業的AI面試官，正在對候選人進行面試。
        面試職位是「{self.job_title}」。
        以下是職位描述的相關資訊，請參考這些資訊來提問：
        {context}

        你的目標是評估候選人的能力，並在適當的時候結束面試。
        如果面試可以結束，請回覆「[面試結束]」作為你的回答，否則請提出下一個面試問題。

        當前對話歷史：
        {{history}}
        候選人: {{input}}
        AI 面試官:"""
        
        PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
        self.conversation.prompt = PROMPT

        # The input to the conversation chain will be an empty string, as the history is managed by memory
        # The actual user input is already added to memory in process_user_answer
        ai_response = await self.conversation.arun(input="") 
        
        if "[面試結束]" in ai_response:
            self.interview_completed = True
            final_message = ai_response.replace("[面試結束]", "").strip()
            if not final_message: # If AI just returned the tag, provide a default message
                final_message = "謝謝您今天來參加面試，面試到此結束。我們將在完成所有評估後通知您結果。"
            
            self.conversation_history.append({"role": "model", "parts": [{"text": final_message}]})
            audio_url = await generate_and_upload_audio(final_message)

            end_time = time.time()
            logging.info(f"[{self.session_id}] 面試已結束，耗時: {end_time - start_time:.2f} 秒。")
            return {"text": final_message, "audio_url": audio_url, "interview_ended": True}
        else:
            next_question_text = ai_response
            self.conversation_history.append({"role": "model", "parts": [{"text": next_question_text}]})
            audio_url = await generate_and_upload_audio(next_question_text)
            
            end_time = time.time()
            logging.info(f"[{self.session_id}] 已準備好下一個問題，耗時: {end_time - start_time:.2f} 秒。")
            return {"text": next_question_text, "audio_url": audio_url, "interview_ended": False}

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

    async def _generate_dynamic_question(self, job_title: str, job_description: str, is_first_question: bool = False) -> str:
        logging.info(f"[{self.session_id}] 開始生成動態面試問題，職位: '{job_title}'。")
        start_time = time.time()

        # Use RAG to retrieve relevant info from job description
        retrieved_docs = []
        if self.retriever:
            # For the first question, query with job title, otherwise use last AI message
            query = job_title if is_first_question else self.memory.chat_memory.messages[-1].content
            retrieved_docs = self.retriever.get_relevant_documents(query)
            logging.info(f"[{self.session_id}] RAG 檢索到 {len(retrieved_docs)} 份相關文件。")
        
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        if is_first_question:
            prompt = f"""你是一位專業的AI面試官。請根據應徵職位「{job_title}」及職位描述「{job_description}」。
            以下是從職位描述中檢索到的相關資訊，請參考這些資訊來設計第一個面試問題：
            {context}

            請提出第一個面試問題。"""
        else:
            # This part is now handled by get_next_question using ConversationChain
            # This function will primarily be used for the initial question generation.
            # If it's called for subsequent questions, it means something went wrong with ConversationChain.
            prompt = f"""你是一位專業的AI面試官。請根據應徵職位「{job_title}」及職位描述「{job_description}」。
            以下是從職位描述中檢索到的相關資訊，請參考這些資訊來設計一個面試問題：
            {context}

            請提出一個面試問題。"""

        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = await call_gemini_api(GEMINI_API_KEY, payload)
            question_text = extract_json_from_gemini_response(response) # Assuming Gemini returns plain text for question
            
            end_time = time.time()
            logging.info(f"[{self.session_id}] 動態面試問題生成完成，耗時: {end_time - start_time:.2f} 秒。")
            return question_text
        except Exception as e:
            logging.error(f"[{self.session_id}] 生成動態問題失敗。錯誤: {e}", exc_info=True)
            # Fallback question
            return f"您好，請簡單自我介紹，並說明您為何對「{job_title}」這個職位感興趣。"

    async def _evaluate_answer(self, user_text: str, emotion_result: Dict[str, Any] = None):
        logging.info(f"[{self.session_id}] 開始評估使用者答案。")
        start_time = time.time()
        # Get the last AI message (the question) from LangChain memory
        question_text = ""
        if self.memory.chat_memory.messages:
            for msg in reversed(self.memory.chat_memory.messages):
                if msg.type == "ai":
                    question_text = msg.content
                    break
        
        if not question_text:
            logging.warning(f"[{self.session_id}] 無法從記憶體中獲取當前問題。")
            question_text = "未知問題" # Fallback

        emotion_info = ""
        if emotion_result:
            emotion_info = f"\n候選人臉部情緒分析結果：{emotion_result}"

        # Use RAG to retrieve relevant info from job description for evaluation context
        retrieved_docs = []
        if self.retriever:
            retrieved_docs = self.retriever.get_relevant_documents(question_text + " " + user_text)
            logging.info(f"[{self.session_id}] RAG 檢索到 {len(retrieved_docs)} 份相關文件用於評估。")
        
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        prompt = f"""作為一個AI面試官，請根據問題「{question_text}」、候選人回答「{user_text}」{emotion_info}。
        以下是從職位描述中檢索到的相關資訊，請參考這些資訊來進行評估：
        {context}

        對以下維度進行1-5分評分: {', '.join(EVALUATION_DIMENSIONS)}。同時，請提供對候選人回答的詳細分析和理由，並綜合考慮其情緒表現。請以JSON格式返回，例如：{{"scores": {{"技術深度": 4, "溝通能力": 5}}, "reasoning": "候選人在技術深度方面表現良好，因為..."}}."""
        
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

