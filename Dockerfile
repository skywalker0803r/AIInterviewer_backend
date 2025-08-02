# 使用官方 Python 映像檔作為基礎映像檔
FROM python:3.10-bullseye

# 設定工作目錄
WORKDIR /app

# 安裝 FFmpeg，Whisper 需要它
# 更新套件列表並安裝 FFmpeg 及其相關工具
RUN DEBIAN_FRONTEND=noninteractive apt-get update

# 安裝 FFmpeg 及其相關工具
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y     ffmpeg     libsm6     libxext6     && rm -rf /var/lib/apt/lists/*

# 將 requirements.txt 複製到工作目錄並安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 將整個 backend 目錄複製到容器中
COPY . .

# 暴露 FastAPI 應用程式運行的埠
EXPOSE 8002

# 定義容器啟動時執行的命令
# 使用 uvicorn 啟動 FastAPI 應用程式
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]