from google.cloud import storage
import logging
import io

storage_client = storage.Client()

async def upload_audio_to_gcs(audio_content: bytes, filename: str, bucket_name: str) -> str:
    """Uploads audio content to GCS and returns the public URL."""
    if not bucket_name:
        logging.error("GCS_BUCKET_NAME is not set. Cannot upload audio to GCS.")
        raise ValueError("GCS_BUCKET_NAME environment variable is not set.")

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"audio/{filename}") # Store in an 'audio' folder within the bucket

    # Upload the audio content
    blob.upload_from_string(audio_content, content_type="audio/mpeg")

    # Make the blob publicly accessible
    blob.make_public()

    public_url = blob.public_url
    logging.info(f"Audio uploaded to GCS: {public_url}")
    return public_url
