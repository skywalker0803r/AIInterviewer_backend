import httpx
import logging
import time

async def get_jobs_from_104(keyword: str = "前端工程師") -> list:
    logging.info(f"開始從 104 搜尋職缺，關鍵字: '{keyword}'")
    start_time_overall = time.time()
    
    url = (
        "https://www.104.com.tw/jobs/search/list"
        f"?ro=0&keyword={keyword}&jobcatExpMore=1&order=11&page=1"
    )
    logging.info(f"請求 104 API URL: {url}")
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.104.com.tw/jobs/search/"
    }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, headers=headers)
            logging.info(f"104 JSON API 回應狀態碼: {resp.status_code}")
            if resp.status_code != 200:
                logging.error(f"104 API 返回非 200 狀態碼: {resp.status_code}")
                return []

            data = resp.json()
            job_list = data.get("data", {}).get("list", [])
            logging.info(f"從 104 API 檢索到 {len(job_list)} 個職缺。")

            result = []
            for job in job_list[:10]:
                job_no = job.get('jobNo')
                relative_job_url = job.get('link', {}).get('job')
                if relative_job_url:
                    job_url = f"https:{relative_job_url}"
                else:
                    job_url = f"https://www.104.com.tw/job/{job_no}"
                result.append({
                    "title": job.get("jobName"),
                    "company": job.get("custName"),
                    "url": job_url,
                    "description": job.get("description", "")
                })
            end_time_overall = time.time()
            logging.info(f"104 職缺搜尋完成，耗時: {end_time_overall - start_time_overall:.2f} 秒。返回 {len(result)} 個職缺。")
            return result
        except httpx.RequestError as e:
            logging.error(f"請求 104 API 時發生網路錯誤: {e}", exc_info=True)
            return []
        except Exception as e:
            logging.error(f"解析 104 API JSON 或處理數據時發生錯誤: {e}", exc_info=True)
            return []
