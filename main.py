# main.py
import os
import re
import json
import base64
import stat
import shutil
import asyncio
import logging
import sys
from typing import List, Optional
from datetime import datetime

import httpx
import git
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# ------------------------- Constants -------------------------
MIT_LICENSE_TEMPLATE = """MIT License

Copyright (c) {year} {fullname}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

# ------------------------- Settings -------------------------
class Settings(BaseSettings):
    GEMINI_API_KEY: str = Field("", env="GEMINI_API_KEY")
    GITHUB_TOKEN: str = Field("", env="GITHUB_TOKEN")
    GITHUB_USERNAME: str = Field("", env="GITHUB_USERNAME")
    STUDENT_SECRET: str = Field("", env="STUDENT_SECRET")
    LOG_FILE_PATH: str = Field("logs/app.log", env="LOG_FILE_PATH")
    MAX_CONCURRENT_TASKS: int = Field(2, env="MAX_CONCURRENT_TASKS")
    KEEP_ALIVE_INTERVAL_SECONDS: int = Field(30, env="KEEP_ALIVE_INTERVAL_SECONDS")
    GITHUB_API_BASE: str = Field("https://api.github.com", env="GITHUB_API_BASE")
    GITHUB_PAGES_BASE: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
if not settings.GITHUB_PAGES_BASE:
    settings.GITHUB_PAGES_BASE = f"https://{settings.GITHUB_USERNAME}.github.io"

# ------------------------- Logging -------------------------
os.makedirs(os.path.dirname(settings.LOG_FILE_PATH), exist_ok=True)
logger = logging.getLogger("task_receiver")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(settings.LOG_FILE_PATH, mode="a", encoding="utf-8")
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(fmt)
file_handler.setFormatter(fmt)
logger.handlers = []
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.propagate = False


def flush_logs():
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        for h in logger.handlers:
            try:
                h.flush()
            except Exception:
                pass
    except Exception:
        pass


# ------------------------- Models -------------------------
class Attachment(BaseModel):
    name: str
    url: str  # data URI or http(s) url


class TaskRequest(BaseModel):
    task: str
    email: str
    round: int
    brief: str
    evaluation_url: str
    nonce: str
    secret: str
    attachments: List[Attachment] = []


# ------------------------- App & Globals -------------------------
app = FastAPI(
    title="Automated Task Receiver & Processor",
    description="LLM-driven code generation and deployment",
)
background_tasks_list: List[asyncio.Task] = []
task_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_TASKS)
last_received_task: Optional[dict] = None
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

# ------------------------- Utility -------------------------
def verify_secret(secret_from_request: str) -> bool:
    return secret_from_request == settings.STUDENT_SECRET


def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def remove_local_path(path: str):
    if not os.path.exists(path):
        return

    def onerror(func, path_arg, exc_info):
        try:
            os.chmod(path_arg, stat.S_IWUSR)
            func(path_arg)
        except Exception as exc:
            logger.exception(f"Failed in rmtree on {path_arg}: {exc}")
            raise

    logger.info(f"[CLEANUP] Removing local directory: {path}")
    shutil.rmtree(path, onerror=onerror)
    flush_logs()


# Basic check for common secret patterns
async def check_for_secrets(local_path: str) -> bool:
    """Basic secret scanning for common patterns"""
    patterns = [
        r'(?i)(api[_-]?key|apikey)["\s:=]+[\w-]{20,}',
        r'(?i)(secret[_-]?key|password)["\s:=]+\w{8,}',
        r'ghp_[a-zA-Z0-9]{36}',  # GitHub tokens
        r'gho_[a-zA-Z0-9]{36}',  # GitHub OAuth tokens
        r'AKIA[0-9A-Z]{16}',     # AWS Access Key
    ]

    try:
        for root, _, files in os.walk(local_path):
            for file in files:
                if file.endswith(('.py', '.js', '.html', '.json', '.env', '.txt', '.md')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            for pattern in patterns:
                                if re.search(pattern, content):
                                    logger.warning(f"[SECRET_SCAN] Potential secret found in {file}")
                                    return True
                    except Exception:
                        pass
        logger.info("[SECRET_SCAN] No secrets detected")
        return False
    except Exception as e:
        logger.warning(f"[SECRET_SCAN] Error during scan: {e}")
        return False


# ------------------------- Attachment helpers -------------------------
def is_image_data_uri(data_uri: str) -> bool:
    if not data_uri or not data_uri.startswith("data:"):
        return False
    return re.search(r"data:image/[^;]+;base64,", data_uri, re.IGNORECASE) is not None


def data_uri_to_gemini_part(data_uri: str) -> Optional[dict]:
    if not data_uri or not data_uri.startswith("data:"):
        return None
    match = re.search(
        r"data:(?P<mime_type>[^;]+);base64,(?P<base64_data>.*)",
        data_uri,
        re.IGNORECASE,
    )
    if not match:
        return None
    mime_type = match.group("mime_type")
    base64_data = match.group("base64_data")
    if not mime_type.startswith("image/"):
        return None
    return {"inlineData": {"data": base64_data, "mimeType": mime_type}}


async def attachment_to_gemini_part(attachment_url: str) -> Optional[dict]:
    if not attachment_url:
        return None
    if attachment_url.startswith("data:"):
        return data_uri_to_gemini_part(attachment_url)
    if attachment_url.startswith(("http://", "https://")):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(attachment_url)
                resp.raise_for_status()
                mime = resp.headers.get("Content-Type", "")
                if not mime.startswith("image/"):
                    logger.info(f"[ATTACHMENT] Skipping non-image MIME: {mime}")
                    return None
                b64 = base64.b64encode(resp.content).decode("utf-8")
                return {"inlineData": {"data": b64, "mimeType": mime}}
        except Exception as e:
            logger.warning(
                f"[ATTACHMENT] Failed to fetch/encode attachment {attachment_url}: {e}"
            )
            return None
    return None


# ------------------------- Filesystem Save Helpers -------------------------
async def save_generated_files_locally(task_id: str, files: dict) -> str:
    base_dir = os.path.join(os.getcwd(), "generated_tasks")
    task_dir = os.path.join(base_dir, task_id)
    safe_makedirs(task_dir)
    logger.info(f"[LOCAL_SAVE] Saving generated files to: {task_dir}")
    for filename, content in files.items():
        file_path = os.path.join(task_dir, filename)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"   -> Saved: {filename} (bytes: {len(content)})")
        except Exception as e:
            logger.exception(f"Failed to save generated file {filename}: {e}")
            raise
    flush_logs()
    return task_dir


async def save_attachments_locally(task_dir: str, attachments: List[Attachment]) -> List[str]:
    saved_files = []
    logger.info(f"[ATTACHMENTS] Processing {len(attachments)} attachments for {task_dir}")
    async with httpx.AsyncClient(timeout=30) as client:
        for attachment in attachments:
            filename = attachment.name
            url = attachment.url
            file_bytes = None
            if not filename or not url:
                logger.warning(f"Skipping invalid attachment entry: {filename}")
                continue
            try:
                if url.startswith("data:"):
                    m = re.search(r"base64,(.*)", url, re.IGNORECASE)
                    if m:
                        file_bytes = base64.b64decode(m.group(1))
                elif url.startswith(("http://", "https://")):
                    resp = await client.get(url)
                    resp.raise_for_status()
                    file_bytes = resp.content
                if file_bytes is None:
                    logger.warning(f"No content for attachment: {filename}")
                    continue
                file_path = os.path.join(task_dir, filename)
                with open(file_path, "wb") as f:
                    f.write(file_bytes)
                logger.info(
                    f"   -> Saved Attachment: {filename} (bytes: {len(file_bytes)})"
                )
                saved_files.append(filename)
            except Exception as e:
                logger.exception(f"Failed to save attachment {filename}: {e}")
    flush_logs()
    return saved_files


# ------------------------- GitHub helpers -------------------------
async def setup_local_repo(
    local_path: str,
    repo_name: str,
    repo_url_auth: str,
    repo_url_http: str,
    round_index: int,
) -> git.Repo:
    github_token = settings.GITHUB_TOKEN
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    async with httpx.AsyncClient(timeout=45) as client:
        try:
            if round_index == 1:
                logger.info(f"[GIT] R1: Creating remote repo '{repo_name}'")
                payload = {"name": repo_name, "private": False, "auto_init": True}
                resp = await client.post(
                    f"{settings.GITHUB_API_BASE}/user/repos",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                repo = git.Repo.init(local_path)
                repo.create_remote("origin", repo_url_auth)
                logger.info("[GIT] Local repo initialized")
            else:
                logger.info(f"[GIT] R{round_index}: Cloning {repo_url_http}")
                repo = git.Repo.clone_from(repo_url_auth, local_path)
                logger.info("[GIT] Cloned repo")
            flush_logs()
            return repo
        except httpx.HTTPStatusError as e:
            logger.exception(f"GitHub API error: {getattr(e.response, 'text', '')}")
            raise
        except git.GitCommandError as e:
            logger.exception(f"Git command error: {e}")
            raise


async def commit_and_publish(
    repo: git.Repo, task_id: str, round_index: int, repo_name: str
) -> dict:
    github_username = settings.GITHUB_USERNAME
    github_token = settings.GITHUB_TOKEN
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    repo_url_http = f"https://github.com/{github_username}/{repo_name}"

    async with httpx.AsyncClient(timeout=45) as client:
        try:
            repo.git.add(A=True)
            commit_message = (
                f"Task {task_id} - Round {round_index}: automated update"
            )
            repo.index.commit(commit_message)
            commit_sha = repo.head.object.hexsha
            logger.info(f"[GIT] Committed: {commit_sha}")
            repo.git.branch("-M", "main")
            repo.git.push("--set-upstream", "origin", "main", force=True)
            logger.info("[GIT] Pushed to origin/main")

            # Configure GitHub Pages with retries
            pages_api_url = (
                f"{settings.GITHUB_API_BASE}/repos/{github_username}/{repo_name}/pages"
            )
            pages_payload = {"source": {"branch": "main", "path": "/"}}
            pages_max_retries = 5
            pages_base_delay = 3
            for attempt in range(pages_max_retries):
                try:
                    pages_response = await client.get(pages_api_url, headers=headers)
                    is_configured = pages_response.status_code == 200
                    if is_configured:
                        await client.put(
                            pages_api_url, json=pages_payload, headers=headers
                        )
                    else:
                        await client.post(
                            pages_api_url, json=pages_payload, headers=headers
                        )
                    logger.info("[GIT] Pages configured")
                    break
                except httpx.HTTPStatusError as e:
                    text = getattr(e.response, "text", "")
                    if (
                        e.response.status_code == 422
                        and "main branch must exist" in text
                        and attempt < pages_max_retries - 1
                    ):
                        delay = pages_base_delay * (2 ** attempt)
                        logger.warning(
                            f"[GIT] Pages timing issue, retrying in {delay}s"
                        )
                        await asyncio.sleep(delay)
                        continue
                    logger.exception(f"[GIT] Pages configuration failed: {text}")
                    raise

            pages_url = f"{settings.GITHUB_PAGES_BASE}/{repo_name}/"
            flush_logs()
            return {
                "repo_url": repo_url_http,
                "commit_sha": commit_sha,
                "pages_url": pages_url,
            }
        except git.GitCommandError:
            logger.exception("Git operation failed during deployment.")
            raise
        except httpx.HTTPStatusError:
            logger.exception("GitHub API error during deployment.")
            raise


# ------------------------- GitHub Pages verification -------------------------
async def verify_pages_accessible(pages_url: str, max_wait: int = 60) -> bool:
    """Verify GitHub Pages is accessible (200 OK)"""
    logger.info(f"[PAGES_VERIFY] Checking accessibility of {pages_url}")
    start_time = asyncio.get_event_loop().time()

    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
        while (asyncio.get_event_loop().time() - start_time) < max_wait:
            try:
                check_resp = await client.get(pages_url)
                if check_resp.status_code == 200:
                    logger.info(f"[PAGES_VERIFY] ✓ Pages accessible at {pages_url}")
                    return True
            except Exception as e:
                logger.debug(f"[PAGES_VERIFY] Attempt failed: {e}")
            await asyncio.sleep(5)

    logger.warning(f"[PAGES_VERIFY] ✗ Pages not accessible after {max_wait}s")
    return False


# ------------------------- Gemini / LLM helpers -------------------------
async def call_gemini_api(
    contents: list,
    system_prompt: str,
    response_schema: dict,
    max_retries: int = 3,
    timeout: int = 60,
) -> dict:
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema,
        },
    }
    base_delay = 1
    for attempt in range(max_retries):
        try:
            if not settings.GEMINI_API_KEY:
                raise Exception("GEMINI_API_KEY not configured.")
            url = f"{GEMINI_API_URL}?key={settings.GEMINI_API_KEY}"
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    url, json=payload, headers={"Content-Type": "application/json"}
                )
                resp.raise_for_status()
                result = resp.json()
                candidates = result.get("candidates", [])
                if not candidates:
                    raise ValueError("No candidates in LLM response")
                content_parts = candidates[0].get("content", {}).get("parts", [])
                if not content_parts:
                    raise ValueError("No content parts in candidate")
                json_text = content_parts[0].get("text")
                return json.loads(json_text)
        except httpx.HTTPStatusError as e:
            logger.warning(f"[GEMINI] HTTP error attempt {attempt+1}: {e}")
        except (httpx.RequestError, KeyError, json.JSONDecodeError, ValueError) as e:
            logger.warning(f"[GEMINI] Processing error attempt {attempt+1}: {e}")
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            logger.info(f"[GEMINI] Retrying in {delay}s...")
            await asyncio.sleep(delay)
    raise Exception("LLM generation failed after retries")


# ------------------------- Round 2 surgical update (Base.py style) -------------------------
async def call_llm_round2_surgical_update(task_id: str, brief: str, existing_index_html: str) -> dict:
    system_prompt = (
        "You are an expert full-stack engineer tasked with making SURGICAL and MINIMAL changes. "
        "Your MOST CRITICAL instruction is to preserve the existing application's core logic and structure. "
        "Only apply the specific changes requested in the 'New Brief'. "
        "Return a JSON object with 'index.html', 'README.md', and 'LICENSE'. "
        "If README.md / LICENSE exist, copy them verbatim unless a change is strictly required."
    )
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "index.html": {"type": "STRING"},
            "README.md": {"type": "STRING"},
            "LICENSE": {"type": "STRING"},
        },
        "required": ["index.html", "README.md", "LICENSE"]
    }
    prompt = (
        f"UPDATE INSTRUCTION (SAFE MODE):\n\n"
        f"New Brief: {brief}\n\n"
        f"--- EXISTING index.html START ---\n{existing_index_html}\n--- EXISTING index.html END ---\n\n"
        "Only make the minimal changes necessary to implement the brief. Do NOT remove or break core scripts, event handlers, or layout. "
        "Return FULL JSON with 'index.html', 'README.md', 'LICENSE'. If you make no changes to README/LICENSE, copy their existing contents."
    )

    contents = [{"parts": [{"text": prompt}]}]

    try:
        result = await call_gemini_api(
            contents=contents, 
            system_prompt=system_prompt, 
            response_schema=response_schema, 
            max_retries=4, 
            timeout=90
        )
    except Exception as e:
        logger.exception(f"[ROUND2] LLM call failed: {e}")
        return {
            "index.html": existing_index_html or "<!-- preserved due to LLM failure -->",
            "README.md": "", 
            "LICENSE": ""
        }

    # Safety checks
    new_html = (result.get("index.html") or "").strip()
    if not new_html:
        logger.warning("[SAFE] LLM returned empty index.html — reverting to existing.")
        result["index.html"] = existing_index_html
    else:
        try:
            orig_len = len(existing_index_html or "")
            new_len = len(new_html)
            if orig_len > 0 and new_len < max(200, int(orig_len * 0.3)):
                logger.warning("[SAFE] LLM index.html too small. Reverting to existing.")
                result["index.html"] = existing_index_html
        except Exception:
            result["index.html"] = existing_index_html

    result["README.md"] = result.get("README.md") or ""
    result["LICENSE"] = result.get("LICENSE") or ""
    return result


# ------------------------- README Generator -------------------------
async def generate_readme_with_live_url(
    task_id: str, 
    brief: str, 
    pages_url: str, 
    repo_url: str,
    attachments: List[Attachment]
) -> str:
    """Generate README with actual live URL"""
    
    attachment_list = "\n".join([f"- {att.name}" for att in attachments]) if attachments else "None"
    
    system_prompt = (
        "You are a technical writer creating a professional README.md for a GitHub repository. "
        "Create a comprehensive, well-structured README with proper markdown formatting."
    )
    
    prompt = f"""Create a professional README.md for this project:

**Project Brief**: {brief}

**Live Application**: {pages_url}
**Repository**: {repo_url}

**Attachments Used**: 
{attachment_list}

Structure the README with these sections:

1. **Project Title** - Create an appropriate title based on the brief
2. **🚀 Live Demo** - Prominent link to {pages_url} with emphasis
3. **📋 Overview** - Explain what the application does (2-3 sentences)
4. **✨ Features** - List 3-5 key features
5. **🛠️ Technologies Used** - HTML5, Tailwind CSS/Bootstrap, JavaScript, any libraries
6. **📦 Setup** - Clone instructions and how to run locally
7. **💡 Usage** - How to use the application (step-by-step)
8. **📁 Project Structure** - Brief explanation of files
9. **📄 License** - Note that it's under MIT License

Make the Live Demo section prominent. Use proper markdown formatting.
"""

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "content": {"type": "STRING"}
        },
        "required": ["content"]
    }
    
    contents = [{"parts": [{"text": prompt}]}]
    
    try:
        result = await call_gemini_api(
            contents=contents,
            system_prompt=system_prompt,
            response_schema=response_schema,
            max_retries=3,
            timeout=60
        )
        return result.get("content", "")
    except Exception as e:
        logger.warning(f"[README] Failed to generate README: {e}")
        # Fallback README
        return f"""# {task_id.replace('-', ' ').title()}

## 🚀 Live Demo

**[View Live Application]({pages_url})**

## 📋 Overview

{brief}

## 📄 License

This project is licensed under the MIT License.
"""


# ------------------------- Notifier -------------------------
async def notify_evaluation_server(
    evaluation_url: str, 
    email: str, 
    task_id: str, 
    round_index: int, 
    nonce: str, 
    repo_url: str, 
    commit_sha: str, 
    pages_url: str
) -> bool:
    payload = {
        "email": email,
        "task": task_id,
        "round": round_index,
        "nonce": nonce,
        "repo_url": repo_url,
        "commit_sha": commit_sha,
        "pages_url": pages_url
    }
    max_retries = 3
    base_delay = 1
    logger.info(f"[NOTIFY] Notifying evaluation server at {evaluation_url}")
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(evaluation_url, json=payload)
                resp.raise_for_status()
                logger.info(f"[NOTIFY] Notification succeeded: {resp.status_code}")
                flush_logs()
                return True
        except httpx.HTTPStatusError as e:
            logger.warning(f"[NOTIFY] HTTP error attempt {attempt+1}: {e}")
        except httpx.RequestError as e:
            logger.warning(f"[NOTIFY] Request error attempt {attempt+1}: {e}")
        if attempt < max_retries - 1:
            await asyncio.sleep(base_delay * (2 ** attempt))
    
    logger.error("[NOTIFY] Failed to notify evaluation server after retries.")
    flush_logs()
    return False


# ------------------------- Main orchestration -------------------------
async def generate_files_and_deploy(task_data: TaskRequest):
    acquired = False
    start_time = datetime.utcnow()
    
    try:
        await task_semaphore.acquire()
        acquired = True
        logger.info(f"[PROCESS START] Task: {task_data.task} Round: {task_data.round}")
        flush_logs()

        task_id = task_data.task
        email = task_data.email
        round_index = task_data.round
        brief = task_data.brief
        evaluation_url = task_data.evaluation_url
        nonce = task_data.nonce
        attachments = task_data.attachments or []

        # Stable repo name per (task, email) so Round 2 updates the same repo
        safe_task = re.sub(r"[^a-zA-Z0-9\-]+", "-", task_id.lower()).strip("-")
        safe_email = re.sub(r"[^a-zA-Z0-9\-]+", "-", email.split("@")[0].lower()).strip("-")
        repo_name = f"{safe_task}-{safe_email}"[:80]
        
        github_username = settings.GITHUB_USERNAME
        github_token = settings.GITHUB_TOKEN
        repo_url_auth = f"https://{github_username}:{github_token}@github.com/{github_username}/{repo_name}.git"
        repo_url_http = f"https://github.com/{github_username}/{repo_name}"

        base_dir = os.path.join(os.getcwd(), "generated_tasks")
        local_path = os.path.join(base_dir, task_id)

        if round_index == 1:
            # Round 1: Clean slate - remove existing folder and start fresh
            if os.path.exists(local_path):
                try:
                    remove_local_path(local_path)
                    logger.info(f"[CLEANUP] Round 1: Removed existing directory: {local_path}")
                except Exception as e:
                    logger.exception(f"[CLEANUP] Round 1 cleanup failed for {local_path}: {e}")
                    raise
            safe_makedirs(local_path)
            
            # Setup repo: create remote + init local
            repo = await setup_local_repo(local_path, repo_name, repo_url_auth, repo_url_http, round_index)

        else:
            # Round 2+: Reuse existing local repo (no cleanup needed)
            safe_makedirs(local_path)  # Ensure it exists
            
            try:
                logger.info(f"[GIT] Round {round_index}: Reusing existing repo at {local_path}")
                repo = git.Repo(local_path)
                
                # Pull latest changes from remote
                repo.remotes.origin.pull('main')
                logger.info("[GIT] ✓ Pulled latest changes from remote")
                
            except git.InvalidGitRepositoryError:
                # Fallback: If local repo is corrupted/missing, clone fresh
                logger.warning(f"[GIT] Round {round_index}: Local repo invalid/missing, cloning fresh")
                
                # Only NOW try to cleanup if needed
                if os.path.exists(local_path):
                    try:
                        remove_local_path(local_path)
                    except Exception as e:
                        logger.warning(f"[CLEANUP] Fallback cleanup failed, continuing anyway: {e}")
                
                safe_makedirs(local_path)
                repo = await setup_local_repo(local_path, repo_name, repo_url_auth, repo_url_http, round_index)
        # --- Prepare attachment data for LLM ---
        image_parts = []
        for attachment in attachments:
            part = await attachment_to_gemini_part(attachment.url)
            if part:
                image_parts.append(part)

        # Build explicit file reference list for LLM
        attachment_descriptions = ""
        if attachments:
            attachment_descriptions = "\nThe following attachments are provided (saved locally, use exact filenames):\n"
            for att in attachments:
                file_type = att.name.split('.')[-1] if '.' in att.name else 'unknown'
                attachment_descriptions += f"- {att.name} (type: {file_type})\n"
            attachment_descriptions += (
                "IMPORTANT: Reference these files using their exact names in your HTML "
                "(e.g., <img src='sample.png'>). Do NOT rename or use external URLs.\n"
            )

        # --- Round 1: Full generation ---
        if round_index == 1:
            logger.info("[WORKFLOW] Round 1: full generation")

            enriched_brief = f"{brief}\n\n{attachment_descriptions}".strip()

            system_prompt = (
                "You are an expert full-stack engineer. Produce a JSON object with keys 'index.html', 'README.md', and 'LICENSE'. "
                "index.html must be a single-file responsive HTML app using Tailwind CSS from CDN. "
                "CRITICAL: Attachments are saved as local files. Reference them using their EXACT filenames "
                "(e.g., <img src='sample.png'> NOT external URLs). "
                "\n\n"
                "README.md MUST include:\n"
                "1. Project Title\n"
                "2. Live Demo section with placeholder 'DEPLOYMENT_URL_PLACEHOLDER'\n"
                "3. Overview, Features, Technologies Used, Setup, Usage, Project Structure, License\n"
                "\n"
                "LICENSE must contain the complete MIT license text with copyright year and name placeholders."
            )

            response_schema = {
                "type": "OBJECT",
                "properties": {
                    "index.html": {"type": "STRING"},
                    "README.md": {"type": "STRING"},
                    "LICENSE": {"type": "STRING"},
                },
                "required": ["index.html", "README.md", "LICENSE"],
            }

            contents = []
            if image_parts:
                contents.append({"parts": image_parts + [{"text": enriched_brief}]})
            else:
                contents.append({"parts": [{"text": enriched_brief}]})

            generated = await call_gemini_api(
                contents=contents,
                system_prompt=system_prompt,
                response_schema=response_schema,
                max_retries=4,
                timeout=120,
            )
            
            # Validate generated files
            if not generated.get("index.html"):
                raise Exception("LLM failed to generate index.html")
            
            # Ensure MIT license
            current_year = datetime.utcnow().year
            if "MIT License" not in generated.get("LICENSE", ""):
                generated["LICENSE"] = MIT_LICENSE_TEMPLATE.format(
                    year=current_year,
                    fullname="Project Author"
                )

        # --- Round 2+: Surgical Update ---
        else:
            logger.info("[WORKFLOW] Round 2+: surgical update")
            existing_index_html = ""
            idx_path = os.path.join(local_path, "index.html")
            if os.path.exists(idx_path):
                try:
                    with open(idx_path, "r", encoding="utf-8") as f:
                        existing_index_html = f.read()
                    logger.info("[WORKFLOW] Read existing index.html for context")
                except Exception as e:
                    logger.warning(f"[WORKFLOW] Could not read existing index.html: {e}")

            brief_with_attachments = f"{brief}\n\n{attachment_descriptions}".strip()
            generated = await call_llm_round2_surgical_update(
                task_id=task_id, 
                brief=brief_with_attachments, 
                existing_index_html=existing_index_html
            )

            # Preserve README/LICENSE if LLM didn't return them
            readme_path = os.path.join(local_path, "README.md")
            license_path = os.path.join(local_path, "LICENSE")
            if not generated.get("README.md") and os.path.exists(readme_path):
                with open(readme_path, "r", encoding="utf-8") as f:
                    generated["README.md"] = f.read()
            if not generated.get("LICENSE") and os.path.exists(license_path):
                with open(license_path, "r", encoding="utf-8") as f:
                    generated["LICENSE"] = f.read()

        # Save generated files locally
        await save_generated_files_locally(task_id, generated)

        # Save attachments into repo folder (Round 1 or if new attachments in Round 2)
        if round_index == 1 or attachments:
            await save_attachments_locally(local_path, attachments)
        
        # Check for secrets before committing
        has_secrets = await check_for_secrets(local_path)
        if has_secrets:
            logger.error("[SECRET_SCAN] Secrets detected! Aborting deployment.")
            raise Exception("Secrets found in generated files")

        # Commit and publish
        deployment_info = await commit_and_publish(repo, task_id, round_index, repo_name)
        
        # Verify Pages is accessible
        await verify_pages_accessible(deployment_info["pages_url"], max_wait=60)
        
        # Generate and update README with live URL (Round 1 only)
        if round_index == 1:
            logger.info("[README] Generating README with live URL")
            try:
                updated_readme = await generate_readme_with_live_url(
                    task_id=task_id,
                    brief=brief,
                    pages_url=deployment_info["pages_url"],
                    repo_url=deployment_info["repo_url"],
                    attachments=attachments
                )
                
                readme_path = os.path.join(local_path, "README.md")
                with open(readme_path, "w", encoding="utf-8") as f:
                    f.write(updated_readme)
                
                # Commit the updated README
                repo.git.add("README.md")
                repo.index.commit("Update README with live demo link")
                repo.git.push('origin', 'main')
                logger.info("[README] ✓ Updated and pushed README with live URL")
            except Exception as e:
                logger.warning(f"[README] Failed to update with live URL: {e}")

        # Check timing (10-minute deadline)
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        if elapsed > 600:
            logger.warning(f"[TIMING] ⚠ Exceeded 10-minute deadline: {elapsed:.1f}s")
        else:
            logger.info(f"[TIMING] ✓ Completed in {elapsed:.1f}s (under 10min deadline)")

        # Notify evaluation server
        await notify_evaluation_server(
            evaluation_url=evaluation_url,
            email=email,
            task_id=task_id,
            round_index=round_index,
            nonce=nonce,
            repo_url=deployment_info["repo_url"],
            commit_sha=deployment_info["commit_sha"],
            pages_url=deployment_info["pages_url"],
        )

        logger.info(f"[DEPLOYMENT] ✓ Success. Repo: {deployment_info['repo_url']} Pages: {deployment_info['pages_url']}")

    except Exception as exc:
        logger.exception(f"[CRITICAL FAILURE] Task {getattr(task_data, 'task', 'unknown')} failed: {exc}")
    finally:
        if acquired:
            task_semaphore.release()
        flush_logs()
        logger.info(f"[PROCESS END] Task: {getattr(task_data, 'task', 'unknown')} Round: {getattr(task_data, 'round', 'unknown')}")


# ------------------------- Endpoint handlers -------------------------
def _task_done_callback(task: asyncio.Task):
    try:
        exc = task.exception()
        if exc:
            logger.error(f"[BACKGROUND TASK] Task finished with exception: {exc}")
            logger.exception(exc)
        else:
            logger.info("[BACKGROUND TASK] Task finished successfully.")
    except asyncio.CancelledError:
        logger.warning("[BACKGROUND TASK] Task was cancelled.")
    finally:
        flush_logs()


@app.post("/ready", status_code=200)
async def receive_task(task_data: TaskRequest, request: Request):
    global last_received_task, background_tasks_list
    if not verify_secret(task_data.secret):
        logger.warning(
            f"Unauthorized attempt for task {task_data.task} from "
            f"{request.client.host if request.client else 'unknown'}"
        )
        raise HTTPException(status_code=401, detail="Unauthorized: Secret mismatch")

    last_received_task = {
        "task": task_data.task,
        "email": task_data.email,
        "round": task_data.round,
        "brief": (task_data.brief[:250] + "...") if len(task_data.brief) > 250 else task_data.brief,
        "time": datetime.utcnow().isoformat() + "Z"
    }

    bg_task = asyncio.create_task(generate_files_and_deploy(task_data))
    bg_task.add_done_callback(_task_done_callback)
    background_tasks_list.append(bg_task)

    logger.info(f"Received task {task_data.task}. Background processing started.")
    flush_logs()

    return JSONResponse(
        status_code=200, 
        content={
            "status": "ready", 
            "message": f"Task {task_data.task} received and processing started."
        }
    )


@app.get("/")
async def root():
    return {"message": "Task Receiver Service running. POST /ready to submit."}


@app.get("/status")
async def get_status():
    global last_received_task, background_tasks_list
    if last_received_task:
        background_tasks_list[:] = [t for t in background_tasks_list if not t.done()]
        return {
            "last_received_task": last_received_task, 
            "running_background_tasks": len(background_tasks_list)
        }
    return {"message": "Awaiting first task submission to /ready"}


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"}


@app.get("/logs")
async def get_logs(lines: int = Query(200, ge=1, le=5000)):
    path = settings.LOG_FILE_PATH
    if not os.path.exists(path):
        return PlainTextResponse("Log file not found.", status_code=404)
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            buffer = bytearray()
            block_size = 1024
            blocks = 0
            while file_size > 0 and len(buffer) < lines * 2000 and blocks < 1024:
                read_size = min(block_size, file_size)
                f.seek(file_size - read_size)
                buffer.extend(f.read(read_size))
                file_size -= read_size
                blocks += 1
            text = buffer.decode(errors="ignore").splitlines()
            last_lines = "\n".join(text[-lines:])
            return PlainTextResponse(last_lines)
    except Exception as e:
        logger.exception(f"Error reading log file: {e}")
        return PlainTextResponse(f"Error reading log file: {e}", status_code=500)


# ------------------------- Startup / Shutdown -------------------------
@app.on_event("startup")
async def startup_event():
    async def keep_alive():
        while True:
            try:
                logger.info("[KEEPALIVE] Service heartbeat")
                flush_logs()
            except Exception:
                pass
            await asyncio.sleep(settings.KEEP_ALIVE_INTERVAL_SECONDS)
    asyncio.create_task(keep_alive())


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("[SHUTDOWN] Waiting for background tasks to finish (graceful shutdown)...")
    for t in background_tasks_list:
        if not t.done():
            try:
                t.cancel()
            except Exception:
                pass
    await asyncio.sleep(0.5)
    flush_logs()

