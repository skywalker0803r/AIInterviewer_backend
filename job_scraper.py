import httpx
import logging

async def get_jobs_from_104(keyword: str = "前端工程師") -> list:
    logging.info(f"Receiving job search keyword: {keyword}")
    
    url = (
        "https://www.104.com.tw/jobs/search/list"
        f"?ro=0&keyword={keyword}&jobcatExpMore=1&order=11&page=1"
    )
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.104.com.tw/jobs/search/"
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers)
        logging.info(f"104 JSON API response status code: {resp.status_code}")
        if resp.status_code != 200:
            return []

        try:
            data = resp.json()
            job_list = data.get("data", {}).get("list", [])
            logging.info(f"Number of jobs retrieved: {len(job_list)}")

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
            return result
        except Exception as e:
            logging.error(f"Error parsing JSON from 104 API: {e}")
            return []
