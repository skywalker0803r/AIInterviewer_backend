import httpx
import logging
import json

async def call_gemini_api(api_key: str, payload: dict) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    timeout = httpx.Timeout(30.0, read=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()
        except httpx.ReadTimeout:
            logging.error("連線到 Gemini API 時讀取超時")
            return None
        except Exception as e:
            logging.error(f"呼叫 Gemini API 發生錯誤: {e}")
            return None

def extract_json_from_gemini_response(gemini_reply: dict) -> str:
    if "candidates" not in gemini_reply or not gemini_reply["candidates"]:
        logging.error(f"Gemini API did not return candidates or candidates list is empty: {json.dumps(gemini_reply, ensure_ascii=False, indent=2)}")
        raise ValueError("API 回應中缺少 candidates 或 candidates 為空，請檢查 API key 和參數")

    response_text = gemini_reply["candidates"][0]["content"]["parts"][0]["text"]
    
    # Extract JSON string from markdown code block
    import re
    json_match = re.search(r"```json\n([\s\S]*?)\n```", response_text)
    if json_match:
        extracted_json = json_match.group(1)
        # 移除所有無效的控制字元，確保 JSON 能夠被解析
        cleaned_json = re.sub(r'[\x00-\x1F]', '', extracted_json)
        return cleaned_json
    else:
        return response_text # Fallback if no markdown block

