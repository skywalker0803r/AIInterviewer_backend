import os
import logging
from google.cloud import storage
from google.oauth2 import service_account

# 判斷是否在 Cloud Run 環境（Cloud Run 有 K_SERVICE 這環境變數）
IS_CLOUD_RUN = os.environ.get("K_SERVICE") is not None

if IS_CLOUD_RUN:
    # Cloud Run 內用預設認證
    storage_client = storage.Client()
else:
    # 本地端需要用金鑰檔案
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        raise EnvironmentError("本地端請設定 GOOGLE_APPLICATION_CREDENTIALS 指向金鑰 JSON 檔案")
    if not os.path.isfile(credentials_path):
        raise FileNotFoundError(f"找不到金鑰檔案：{credentials_path}")
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    storage_client = storage.Client(credentials=credentials)

async def upload_audio_to_gcs(audio_content: bytes, filename: str, bucket_name: str) -> str:
    """Uploads audio content to GCS and returns the public URL."""
    if not bucket_name:
        logging.error("GCS_BUCKET_NAME is not set. Cannot upload audio to GCS.")
        raise ValueError("GCS_BUCKET_NAME environment variable is not set.")

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"audio/{filename}")  # 放在 bucket 裡的 audio 資料夾下

    # 上傳音訊內容
    blob.upload_from_string(audio_content, content_type="audio/mpeg")

    # 設定公開權限
    blob.make_public()

    public_url = blob.public_url
    logging.info(f"Audio uploaded to GCS: {public_url}")
    return public_url
