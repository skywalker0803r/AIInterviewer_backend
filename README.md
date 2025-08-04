# AI 面試官模擬器 - 專案規格書

## 1. 專案總覽 (Project Overview)

AI 面試官模擬器是一個互動式平台，旨在為使用者提供模擬面試練習。該系統允許使用者搜尋職缺，並根據選定的職位描述，由 AI 面試官生成客製化的面試問題。面試過程支援語音互動，並在結束後提供一份綜合性的評估報告，涵蓋多個維度（如技術深度、溝通能力等）。

## 2. 後端規格 (Backend Specification)

### 2.1. 技術棧 (Technology Stack)

*   **程式語言**: Python 3.x
*   **Web 框架**: FastAPI
*   **容器化**: Docker, Docker Compose
*   **AI/ML 服務**: Google Gemini API (用於面試問題生成、面試評估), Whisper (用於語音轉文字), gTTS (Google Text-to-Speech, 用於文字轉語音)
*   **雲端儲存**: Google Cloud Storage (GCS) (透過 `gcs_utils.py` 處理)

### 2.2. 核心模組與功能 (Core Modules and Functionalities)

以下是 `AIInterviewer_backend` 目錄下的主要 Python 模組及其功能描述：

*   **`main.py`**:
    *   專案的入口點，初始化 FastAPI 應用。
    *   定義主要的 API 端點 (HTTP)。
    *   協調各模組之間的互動，處理面試流程的邏輯。
    *   提供職缺搜尋、面試啟動、問題問答、音訊處理和報告生成的接口。
    *   負責靜態檔案（TTS 生成的音訊）的服務。

*   **`config.py`**:
    *   管理應用程式的配置設定，例如 API 金鑰、GCS 儲存桶名稱等。
    *   通常從 `.env` 文件加載環境變數，確保敏感資訊的安全管理。

*   **`job_scraper.py`**:
    *   負責從外部來源（例如 104 人力銀行）爬取或透過 API 獲取職缺資訊。
    *   提供職缺搜尋功能，供前端展示。

*   **`gemini_api.py`**:
    *   封裝與 Google Gemini API 的互動邏輯。
    *   用於生成面試問題（根據職位描述和預設評估維度）。
    *   用於對使用者的面試回答進行評估，生成多維度的評分和反饋。

*   **`speech_to_text.py`**:
    *   實現語音轉文字 (STT) 功能。
    *   使用 Whisper 模型或其他 STT 服務將接收到的音訊數據轉換為文字。
    *   處理音訊輸入的預處理和格式轉換。

*   **`text_to_speech.py`**:
    *   實現文字轉語音 (TTS) 功能。
    *   使用 gTTS 將 AI 面試官的文字回覆轉換為音訊檔案。
    *   負責將生成的音訊檔案儲存到本地靜態目錄或 GCS。

*   **`emotion_analysis.py`**:
    *   （根據 `README.md` 描述，此功能目前可能被註釋掉或未完全啟用）
    *   旨在分析視訊串流中的情緒，為面試評估提供額外維度。
    *   如果啟用，將處理視訊數據並提取情緒特徵。

*   **`gcs_utils.py`**:
    *   提供與 Google Cloud Storage (GCS) 互動的工具函數。
    *   用於上傳、下載或管理儲存在 GCS 上的檔案，例如 TTS 生成的音訊檔案。

*   **`interview_manager.py`**:
    *   管理單個面試會話的狀態和邏輯。
    *   追蹤面試進度、當前問題、使用者回答歷史。
    *   協調 `gemini_api.py` 進行問題生成和評估。
    *   負責生成最終的面試報告。

### 2.3. API 與通訊 (API and Communication)

*   **HTTP API**:
    *   用於初始的職缺搜尋請求 (`/jobs`)。
    *   用於啟動面試會話 (`/start_interview`)。
    *   用於提交使用者答案並獲取下一個問題 (`/submit_answer_and_get_next_question`)，此端點接收音訊檔案和圖像數據。
    *   用於獲取面試報告 (`/get_interview_report`)。
    *   用於結束面試會話 (`/end_interview`)。
    *   基於 FastAPI 的 RESTful 設計。

### 2.4. 資料流 (Data Flow)

1.  **職缺搜尋**: 前端發送 HTTP GET 請求 (`/jobs`) -> `main.py` -> `job_scraper.py` 獲取職缺 -> `main.py` 返回職缺列表給前端。
2.  **啟動面試**: 前端發送 HTTP POST 請求 (`/start_interview`，包含職位描述) -> `main.py` -> `interview_manager.py` 初始化會話並調用 `gemini_api.py` 生成首個問題 -> `main.py` 返回首個問題給前端。
3.  **面試互動 (HTTP POST)**:
    *   使用者語音 -> 前端錄音 -> 透過 HTTP POST 請求 (`/submit_answer_and_get_next_question`) 發送音訊數據和圖像數據 -> `main.py` -> `speech_to_text.py` 轉錄為文字。
    *   轉錄文字 -> `main.py` -> `interview_manager.py` 處理回答並調用 `gemini_api.py` 生成下一個問題或評估。
    *   AI 回覆文字 -> `main.py` -> `text_to_speech.py` 生成音訊檔案並儲存（可能到 GCS 或本地靜態目錄）。
    *   AI 回覆文字和音訊檔案路徑 -> `main.py` 返回給前端。
4.  **面試結束與報告**: 面試結束信號 -> `main.py` -> `interview_manager.py` 生成最終報告 -> `main.py` 返回報告給前端。

### 2.5. 依賴管理 (Dependency Management)

*   所有 Python 依賴項都列在 `requirements.txt` 文件中。
*   Docker 映像構建過程中會安裝這些依賴。

### 2.6. 部署與容器化 (Deployment and Containerization)

*   **`Dockerfile`**:
    *   定義了後端服務的 Docker 映像構建過程。
    *   基於 Python 基礎映像，安裝 `requirements.txt` 中的依賴。
    *   複製應用程式碼，並設定啟動命令。
*   **`docker-compose.yml`**:
    *   定義了多容器 Docker 應用程式的服務。
    *   包含後端服務的配置，例如埠映射、環境變數、卷掛載等。
    *   簡化了開發和部署過程，允許一鍵啟動整個後端環境。

## 3. 前端規格 (Frontend Specification)

### 3.1. 技術棧 (Technology Stack)

*   **標記語言**: HTML5
*   **樣式**: CSS3 (使用 TailwindCSS 框架進行快速樣式開發)
*   **腳本語言**: JavaScript (使用 jQuery 庫簡化 DOM 操作和 AJAX 請求)

### 3.2. 使用者介面與互動 (User Interface and Interaction)

*   **職缺搜尋介面**:
    *   提供輸入框供使用者輸入職缺關鍵字。
    *   顯示從後端獲取的職缺列表，包含職位名稱、公司等資訊。
    *   允許使用者選擇感興趣的職位以啟動面試。
*   **面試互動介面**:
    *   **聊天視窗**: 實時顯示 AI 面試官和使用者的對話內容（文字）。
    *   **音訊播放**: 播放 AI 面試官的語音回覆。
    *   **麥克風控制**: 提供按鈕控制麥克風錄音的開始和結束。
    *   **視訊顯示**: 預留視訊串流顯示區域（儘管情緒辨識功能可能未完全啟用）。
    *   **狀態指示**: 顯示錄音狀態、連接狀態等。
*   **面試報告展示介面**:
    *   面試結束後，顯示從後端獲取的綜合評估報告。
    *   報告可能包含總體分數和各維度（如技術深度、溝通能力）的詳細評分和反饋。

### 3.3. 與後端通訊 (Communication with Backend)

*   **AJAX (jQuery)**:
    *   用於所有與後端的通訊，包括職缺搜尋 (`/jobs`)、啟動面試 (`/start_interview`)、提交使用者答案並獲取下一個問題 (`/submit_answer_and_get_next_question`)、獲取面試報告 (`/get_interview_report`) 和結束面試 (`/end_interview`)。
    *   處理非同步數據交換，更新頁面內容。
    *   音訊數據透過 HTTP POST 請求以 `FormData` 形式發送。

### 3.4. 依賴 (Dependencies)

*   **jQuery**: 用於簡化 DOM 操作、事件處理和 AJAX 請求。
*   **TailwindCSS**: 作為 CSS 框架，用於快速構建響應式和美觀的使用者介面。
*   **原生 Web API**: 使用 `MediaRecorder` 進行麥克風音訊錄製。

## 4. 設定與運行 (Setup and Execution)

### 4.1. 環境變數設定

在 `AIInterviewer_backend` 資料夾中創建一個名為 `.env` 的文件，並添加以下內容：

```
GEMINI_API_KEY=你的Gemini API Key
```
請將 `你的Gemini API Key` 替換為您從 Google Cloud 獲取的實際 Gemini API 金鑰。

### 4.2. 啟動後端服務

1.  打開終端機或命令提示字元。
2.  導航到 `AIInterviewer/AIInterviewer_backend` 目錄：
    ```bash
    cd C:/Users/ricky/Desktop/AIInterviewer/AIInterviewer_backend
    ```
3.  使用 Docker Compose 構建並啟動後端服務：
    ```bash
    docker-compose up --build
    ```
    這將會構建 Docker 映像（如果尚未構建）並啟動 FastAPI 應用程式。服務將在 Docker 容器中運行，通常監聽 8000 埠（可在 `docker-compose.yml` 中查看）。

### 4.3. 開啟前端介面

1.  在您偏好的網頁瀏覽器中，直接打開 `AIInterviewer/AIInterviewer_frontend/index.html` 檔案。
    *   檔案路徑範例：`file:///C:/Users/ricky/Desktop/AIInterviewer/AIInterviewer_frontend/index.html`

現在，您就可以在瀏覽器中體驗 AI 面試官模擬器了。
可以進行RAG跟上下文管理