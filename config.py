import os
from dotenv import load_dotenv

load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BACKEND_PUBLIC_URL = os.getenv("BACKEND_PUBLIC_URL")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

EVALUATION_DIMENSIONS = [
    "技術深度", "領導能力", "溝通能力", "抗壓能力",
    "解決問題能力", "學習能力", "團隊合作", "創新思維"
]

# Suppress gTTS deprecation warning
import warnings
warnings.filterwarnings("ignore", message="'zh-TW' has been deprecated, falling back to 'zh-TW'. This fallback will be removed in a future version.", category=UserWarning)

# Model Configurations
AVAILABLE_MODELS = {
    "gemini-2.5-flash": {
        "display_name": "Gemini 2.5 Flash (Google)",
        "langchain_model_id": "gemini-2.5-flash",
        "api_key_env": "GEMINI_API_KEY"
    },
    "gemma-1.1-2b-it": { # Assuming this is the correct model ID for gemma-3-1b-it
        "display_name": "Gemma 1.1 2B IT (Google)",
        "langchain_model_id": "gemma-1.1-2b-it",
        "api_key_env": "GEMINI_API_KEY" # Gemma also uses Gemini API key
    }
    # Add other models here if needed
}

DEFAULT_MODEL = "gemini-2.5-flash"
