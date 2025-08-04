from dotenv import load_dotenv
import os
from huggingface_hub import login
from transformers import pipeline

# 載入.env文件中的環境變數
load_dotenv()

# 從環境變數中獲取Hugging Face Token
hf_token = os.getenv("HF_TOKEN")

# 使用Token登入Hugging Face Hub
if hf_token:
    login(token=hf_token)
else:
    print("警告：未找到HF_TOKEN，請檢查.env文件。")

# 現在你可以繼續使用pipeline了
# 載入模型
pipe = pipeline("text-generation", model="google/gemma-3-1b-it")

# 準備消息
messages = [
    {"role": "user", "content": "你好嗎?"},
]

# 執行pipeline
response = pipe(messages)
print(response)