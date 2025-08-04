# AI 面試官模擬器 - 專案規格書

## 1. 專案總覽 (Project Overview)

AI 面試官模擬器是一個互動式平台，旨在為使用者提供模擬面試練習。該系統允許使用者搜尋職缺，並根據選定的職位描述，由 AI 面試官生成客製化的面試問題。面試過程支援語音互動，並在結束後提供一份綜合性的評估報告，涵蓋多個維度（如技術深度、溝通能力等）。

**核心目標：**
*   提供真實的面試模擬體驗。
*   根據職位描述動態生成面試問題。
*   支援語音輸入和語音回覆。
*   提供多維度的面試評估報告。
*   允許使用者選擇不同的 AI 模型進行面試。

## 2. 後端規格 (Backend Specification)

### 2.1. 技術棧 (Technology Stack)

*   **程式語言**: Python 3.x
*   **Web 框架**: FastAPI
*   **容器化**: Docker, Docker Compose
*   **AI/ML 服務**:
    *   **大型語言模型 (LLM)**: Google Gemini API (用於面試問題生成、面試評估)。支援 `gemini-2.5-flash` 和 `gemma-3-1b-it` 模型。
    *   **語音轉文字 (STT)**: Whisper (透過 `openai-whisper` 庫)。
    *   **文字轉語音 (TTS)**: gTTS (Google Text-to-Speech)。
    *   **情緒分析**: DeepFace (透過 `deepface` 庫，使用 TensorFlow)。
*   **雲端儲存**: Google Cloud Storage (GCS) (透過 `gcs_utils.py` 處理，用於儲存 TTS 生成的音訊檔案)。
*   **狀態管理/緩存**: Redis (用於持久化面試會話狀態，如果 Redis 不可用則回退到記憶體儲存)。
*   **向量資料庫**: FAISS (用於 RAG，儲存職位描述的嵌入向量)。
*   **嵌入模型**: Google Generative AI Embeddings (`models/embedding-001`)。
*   **環境變數管理**: `python-dotenv`。
*   **HTTP 客戶端**: `httpx`。
*   **網頁爬蟲**: `beautifulsoup4` (用於 104 職缺爬取)。
*   **LangChain**: 用於整合 LLM、記憶體管理 (ConversationBufferMemory) 和 RAG (Retrieval-Augmented Generation)。

### 2.2. 核心模組與功能 (Core Modules and Functionalities)

以下是 `AIInterviewer_backend` 目錄下的主要 Python 模組及其功能描述：

*   **`main.py`**:
    *   專案的入口點，初始化 FastAPI 應用。
    *   定義主要的 API 端點 (HTTP)。
    *   協調各模組之間的互動，處理面試流程的邏輯。
    *   提供職缺搜尋、面試啟動、問題問答、音訊處理和報告生成的接口。
    *   負責靜態檔案（TTS 生成的音訊）的服務。
    *   處理 Redis 的會話狀態序列化和反序列化，確保 LangChain 組件在載入時重新初始化。

*   **`config.py`**:
    *   管理應用程式的配置設定，例如 API 金鑰、GCS 儲存桶名稱。
    *   定義了可用的 AI 模型配置 (`AVAILABLE_MODELS`)，包括顯示名稱、LangChain 模型 ID 和對應的 API 金鑰環境變數。
    *   定義了面試評估的維度 (`EVALUATION_DIMENSIONS`)。
    *   通常從 `.env` 文件加載環境變數，確保敏感資訊的安全管理。

*   **`job_scraper.py`**:
    *   負責從外部來源（目前是 104 人力銀行）爬取職缺資訊。
    *   提供職缺搜尋功能，供前端展示。

*   **`gemini_api.py`**:
    *   封裝與 Google Gemini API 的互動邏輯。
    *   用於向 Gemini 模型發送請求並接收回應。
    *   包含 `extract_json_from_gemini_response` 函數，用於從 Gemini 的回應中提取並清理 JSON 數據，處理潛在的控制字元問題。

*   **`speech_to_text.py`**:
    *   實現語音轉文字 (STT) 功能。
    *   使用 Whisper 模型將接收到的音訊數據轉換為文字。
    *   處理音訊輸入的預處理和格式轉換。

*   **`text_to_speech.py`**:
    *   實現文字轉語音 (TTS) 功能。
    *   使用 gTTS 將 AI 面試官的文字回覆轉換為音訊檔案。
    *   負責將生成的音訊檔案上傳到 Google Cloud Storage (GCS)。

*   **`emotion_analysis.py`**:
    *   旨在分析視訊串流中的情緒。
    *   使用 DeepFace 庫處理圖像數據並提取情緒特徵。
    *   目前在 `interview_manager.py` 中，情緒分析結果會被納入評估提示詞，但其結果的影響程度取決於 LLM 的解釋。

*   **`gcs_utils.py`**:
    *   提供與 Google Cloud Storage (GCS) 互動的工具函數。
    *   用於上傳、下載或管理儲存在 GCS 上的檔案，例如 TTS 生成的音訊檔案。
    *   支援 Cloud Run 環境下的預設認證和本地開發環境下的服務帳戶金鑰認證。

*   **`interview_manager.py`**:
    *   管理單個面試會話的狀態和邏輯。
    *   **動態問題生成**: 根據職位描述和對話歷史，動態生成面試問題，而非使用固定問題列表。
    *   **LangChain 整合**:
        *   使用 `ChatGoogleGenerativeAI` 作為 LLM。
        *   使用 `ConversationBufferMemory` 管理對話歷史。
        *   使用 `ConversationChain` 協調 LLM 和記憶體。
        *   使用 `GoogleGenerativeAIEmbeddings` 和 `FAISS` 實現 RAG，將職位描述嵌入並用於檢索相關上下文。
    *   **答案評估**: 呼叫 Gemini API 對使用者回答進行多維度評估，並將情緒分析結果納入考量。
    *   **會話持久化**: 實現 `to_dict()` 和 `from_dict()` 方法，以便將可序列化的會話狀態儲存到 Redis，並在需要時重新載入和重建 LangChain 組件。
    *   **面試結束邏輯**: AI 面試官會根據對話進程和候選人表現，判斷是否結束面試，並透過特定標記 `[面試結束]` 來通知系統。

### 2.3. API 端點與通訊 (API Endpoints and Communication)

後端提供以下 RESTful API 端點：

*   **`GET /jobs`**:
    *   **功能**: 搜尋職缺列表。
    *   **參數**: `keyword` (string, 查詢關鍵字，預設為「前端工程師」)。
    *   **回應**: JSON 格式的職缺列表，包含 `title`, `company`, `url`, `description`。
*   **`POST /start_interview`**:
    *   **功能**: 啟動一個新的面試會話。
    *   **請求體**: JSON 格式，包含 `job` (職缺物件，含 `title`, `description`) 和 `model_name` (string, 選擇的 AI 模型名稱，例如 `gemini-2.5-flash` 或 `gemma-3-1b-it`)。
    *   **回應**: JSON 格式，包含 `session_id` (string, 唯一會話 ID) 和 `first_question` (物件，包含 `text` 和 `audio_url`)。
*   **`POST /submit_answer_and_get_next_question`**:
    *   **功能**: 提交使用者答案（語音和圖像），並獲取 AI 面試官的下一個問題或面試結束通知。
    *   **請求體**: `multipart/form-data` 格式，包含 `session_id` (string)、`audio_file` (file, 使用者語音錄音) 和 `image_data` (string, Base64 編碼的視訊幀圖像數據)。
    *   **回應**: JSON 格式，包含 `text` (AI 回覆文本)、`audio_url` (AI 回覆音訊 URL) 和 `interview_ended` (boolean, 指示面試是否結束)。
*   **`GET /get_interview_report`**:
    *   **功能**: 獲取指定面試會話的綜合評估報告。
    *   **參數**: `session_id` (string, 會話 ID)。
    *   **回應**: JSON 格式的報告，包含 `overall_score` (總體分數)、`dimension_scores` (各維度分數)、`hired` (是否建議錄取) 和 `conversation_history` (對話歷史)。
*   **`POST /end_interview`**:
    *   **功能**: 手動結束一個面試會話並清理資源。
    *   **請求體**: JSON 格式，包含 `session_id` (string)。
    *   **回應**: JSON 格式，包含 `message`。

### 2.4. 資料流 (Data Flow)

1.  **職缺搜尋**:
    *   前端發送 HTTP GET 請求 (`/jobs?keyword=...`)。
    *   `main.py` 接收請求，調用 `job_scraper.py` 獲取職缺。
    *   `main.py` 返回職缺列表給前端。
2.  **啟動面試**:
    *   前端發送 HTTP POST 請求 (`/start_interview`，包含職位描述和選定的 `model_name`)。
    *   `main.py` 接收請求，創建 `InterviewManager` 實例（根據 `model_name` 初始化對應的 LLM）。
    *   `InterviewManager` 將職位描述載入到 FAISS 向量儲存中，並動態生成第一個面試問題。
    *   `InterviewManager` 將第一個問題添加到 LangChain 記憶體和 `conversation_history`。
    *   `text_to_speech.py` 將第一個問題文本轉換為音訊並上傳到 GCS。
    *   `main.py` 將 `InterviewManager` 的可序列化狀態儲存到 Redis。
    *   `main.py` 返回第一個問題文本和音訊 URL 給前端。
3.  **面試互動**:
    *   使用者語音 -> 前端錄音 -> 透過 HTTP POST 請求 (`/submit_answer_and_get_next_question`) 發送音訊數據和圖像數據。
    *   `main.py` 從 Redis 載入會話狀態，並使用 `InterviewManager.from_dict()` 重新構建 `InterviewManager` 實例。
    *   `speech_to_text.py` 將使用者音訊轉錄為文字。
    *   `emotion_analysis.py` 分析圖像數據中的情緒（如果可用）。
    *   `InterviewManager` 將使用者回答添加到 LangChain 記憶體和 `conversation_history`。
    *   `InterviewManager` 調用 `_evaluate_answer` 評估使用者回答（使用 RAG 上下文和情緒分析結果）。
    *   `InterviewManager` 調用 `get_next_question`，使用 LangChain `ConversationChain` 和 RAG 動態生成下一個問題。
    *   如果 AI 決定結束面試（回覆包含 `[面試結束]` 標記），則設置 `interview_completed` 為 True。
    *   `text_to_speech.py` 將 AI 回覆文本轉換為音訊並上傳到 GCS。
    *   `main.py` 將更新後的 `InterviewManager` 狀態儲存回 Redis。
    *   `main.py` 返回 AI 回覆文本、音訊 URL 和面試結束狀態給前端。
4.  **面試結束與報告**:
    *   如果面試結束，前端發送 HTTP GET 請求 (`/get_interview_report?session_id=...`)。
    *   `main.py` 從 Redis 載入會話狀態，重新構建 `InterviewManager` 實例。
    *   `InterviewManager` 生成最終的面試報告。
    *   `main.py` 返回報告給前端。
    *   使用者也可以手動發送 `POST /end_interview` 請求來結束會話並清理 Redis 中的數據。

### 2.5. 依賴管理 (Dependency Management)

所有 Python 依賴項都列在 `requirements.txt` 文件中。Docker 映像構建過程中會安裝這些依賴。

**`requirements.txt` 內容：**
```
torch
tensorflow-cpu
fastapi
uvicorn[standard]
httpx
beautifulsoup4
python-dotenv
gTTS
openai-whisper
deepface
tf_keras
google-cloud-storage
python-multipart
google-auth
transformers 
huggingface_hub
redis
langchain
langchain-google-genai
sentence-transformers
faiss-cpu
langchain-community
```

### 2.6. 部署與容器化 (Deployment and Containerization)

*   **`Dockerfile`**:
    *   定義了後端服務的 Docker 映像構建過程。
    *   基於 Python 基礎映像，安裝 `requirements.txt` 中的依賴。
    *   複製應用程式碼，並設定啟動命令。
*   **`docker-compose.yml`**:
    *   定義了多容器 Docker 應用程式的服務。
    *   包含後端服務的配置，例如埠映射、環境變數、卷掛載等。
    *   簡化了開發和部署過程，允許一鍵啟動整個後端環境。
    *   包含 Redis 服務的配置，用於會話狀態持久化。

## 3. 前端規格 (Frontend Specification)

### 3.1. 技術棧 (Technology Stack)

*   **標記語言**: HTML5
*   **樣式**: CSS3 (使用 TailwindCSS 框架進行快速樣式開發)
*   **腳本語言**: JavaScript (使用 jQuery 庫簡化 DOM 操作和 AJAX 請求)
*   **原生 Web API**: `MediaRecorder` (用於麥克風音訊錄製), `navigator.mediaDevices.getUserMedia` (用於獲取麥克風和攝影機權限)。

### 3.2. 使用者介面與互動 (User Interface and Interaction)

*   **職缺搜尋介面**:
    *   提供輸入框供使用者輸入職缺關鍵字。
    *   顯示從後端獲取的職缺列表，包含職位名稱、公司等資訊。
    *   允許使用者選擇感興趣的職位以啟動面試。
*   **AI 模型選擇**:
    *   提供下拉選單，讓使用者選擇要使用的 AI 模型（例如 Gemini 2.5 Flash 或 Gemma 3.1B IT）。
*   **面試互動介面**:
    *   **聊天視窗**: 實時顯示 AI 面試官和使用者的對話內容（文字）。
    *   **音訊播放**: 播放 AI 面試官的語音回覆。
    *   **麥克風控制**: 提供按鈕控制麥克風錄音的開始和結束。
    *   **視訊顯示**: 顯示使用者攝影機的視訊串流（如果權限允許且設備可用）。
    *   **狀態指示**: 顯示錄音狀態、連接狀態等。
*   **面試報告展示介面**:
    *   面試結束後，顯示從後端獲取的綜合評估報告。
    *   報告包含總體分數和各維度（如技術深度、溝通能力）的詳細評分和反饋。
*   **重新開始面試**: 提供按鈕讓使用者重新開始新的面試會話。

### 3.3. 與後端通訊 (Communication with Backend)

*   **AJAX (jQuery)**:
    *   用於所有與後端的通訊，包括職缺搜尋 (`/jobs`)、啟動面試 (`/start_interview`)、提交使用者答案並獲取下一個問題 (`/submit_answer_and_get_next_question`)、獲取面試報告 (`/get_interview_report`) 和結束面試 (`/end_interview`)。
    *   處理非同步數據交換，更新頁面內容。
    *   音訊數據透過 HTTP POST 請求以 `FormData` 形式發送。
    *   圖像數據（Base64 編碼）也透過 `FormData` 發送。

### 3.4. 依賴 (Dependencies)

*   **jQuery**: 用於簡化 DOM 操作、事件處理和 AJAX 請求。
*   **TailwindCSS**: 作為 CSS 框架，用於快速構建響應式和美觀的使用者介面。
*   **原生 Web API**: 使用 `MediaRecorder` 進行麥克風音訊錄製，`navigator.mediaDevices.getUserMedia` 獲取媒體串流。

## 4. 設定與運行 (Setup and Execution)

### 4.1. 環境變數設定

在 `AIInterviewer_backend` 資料夾中創建一個名為 `.env` 的文件，並添加以下內容：

```
GEMINI_API_KEY=你的Gemini API Key
GCS_BUCKET_NAME=你的GCS儲存桶名稱
REDIS=redis://redis:6379/0  # 如果使用 Docker Compose，這是 Redis 服務的預設 URL
```
請將 `你的Gemini API Key` 替換為您從 Google Cloud 獲取的實際 Gemini API 金鑰。
`GCS_BUCKET_NAME` 替換為您在 Google Cloud Storage 中創建的儲存桶名稱。
如果不在 Docker 環境下運行 Redis，請將 `REDIS` 設置為你的 Redis 服務的實際 URL，或留空以使用記憶體儲存。

### 4.2. 啟動後端服務

1.  打開終端機或命令提示字元。
2.  導航到 `AIInterviewer/AIInterviewer_backend` 目錄：
    ```bash
    cd C:/Users/ricky/Desktop/AIInterviewer/AIInterviewer_backend
    ```
3.  使用 Docker Compose 構建並啟動後端服務（包含 Redis 服務）：
    ```bash
    docker-compose up --build
    ```
    這將會構建 Docker 映像（如果尚未構建）並啟動 FastAPI 應用程式和 Redis 服務。後端服務通常監聽 8000 埠（可在 `docker-compose.yml` 中查看）。

### 4.3. 開啟前端介面

1.  在您偏好的網頁瀏覽器中，直接打開 `AIInterviewer/AIInterviewer_frontend/index.html` 檔案。
    *   檔案路徑範例：`file:///C:/Users/ricky/Desktop/AIInterviewer/AIInterviewer_frontend/index.html`

現在，您就可以在瀏覽器中體驗 AI 面試官模擬器了。

---
