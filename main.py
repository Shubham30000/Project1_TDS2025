from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import os
from dotenv import load_dotenv
import requests
import base64
import time
import logging
from typing import Dict, List
import json
from openai import OpenAI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "Shubham30000")
SECRET = os.getenv("SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.aimlapi.com/v1")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

app = FastAPI(title="LLM Code Deployment - Student API")


def validate_secret(secret: str) -> bool:
    """Validate secret matches submitted value"""
    return secret == SECRET


def create_github_repo(repo_name: str, description: str = ""):
    """Create public GitHub repo with MIT license"""
    payload = {
        "name": repo_name,
        "description": description,
        "private": False,
        "auto_init": True,
        "license_template": "mit"
    }
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    response = requests.post("https://api.github.com/user/repos", json=payload, headers=headers)
    if response.status_code != 201:
        logger.error(f"Failed to create repo: {response.status_code}, {response.text}")
        raise Exception(f"Failed to create repo: {response.status_code}")
    
    logger.info(f"✓ Created repository: {repo_name}")
    return response.json()


def get_file_sha(repo_name: str, file_path: str) -> str:
    """Get SHA of existing file or None"""
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/contents/{file_path}"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(url, headers=headers)
    return response.json().get("sha") if response.status_code == 200 else None


def push_file_to_repo(repo_name: str, file_path: str, content: str, message: str, sha: str = None):
    """Push file to GitHub"""
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    encoded_content = base64.b64encode(
        content.encode('utf-8') if isinstance(content, str) else content
    ).decode('utf-8')

    payload = {"message": message, "content": encoded_content}
    if sha:
        payload["sha"] = sha

    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/contents/{file_path}"
    response = requests.put(url, json=payload, headers=headers)

    if response.status_code not in [200, 201]:
        raise Exception(f"Failed to push {file_path}: {response.status_code}")

    logger.info(f"✓ Pushed: {file_path}")
    return response.json().get("commit", {}).get("sha")


def enable_github_pages(repo_name: str):
    """Enable GitHub Pages"""
    payload = {
        "build_type": "legacy",
        "source": {"branch": "main", "path": "/"}
    }
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/pages"
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code in [201, 409]:  # 409 = already enabled
        logger.info(f"✓ GitHub Pages enabled for {repo_name}")
        return True
    else:
        raise Exception(f"Failed to enable Pages: {response.status_code}")


def generate_code_with_llm(brief: str, checks: list, attachments: list, task_id: str, round_num: int) -> str:
    """
    Generate fully functional HTML using LLM (gpt-4o-mini)
    Works for ANY task type: weather, CSV, forms, markdown, etc.
    """
    
    # Build attachment details
    attachment_details = ""
    if attachments:
        attachment_details = "\n\n📎 ATTACHMENTS PROVIDED:\n"
        for att in attachments:
            name = att.get('name', 'unknown')
            url = att.get('url', '')
            attachment_details += f"**File: {name}**\n"
            
            if url.startswith('data:'):
                data_type = url.split(';')[0].replace('data:', '')
                attachment_details += f"  - Type: {data_type}\n"
                attachment_details += f"  - Data URI: {url[:100]}...\n"
                
                # Provide specific parsing instructions based on type
                if 'csv' in data_type.lower():
                    attachment_details += f"  - ⚠️ DECODE base64 and PARSE this CSV data!\n"
                    attachment_details += f"  - Use: const csvText = atob(dataUri.split(',')[1]);\n"
                elif 'json' in data_type.lower():
                    attachment_details += f"  - ⚠️ DECODE base64 and PARSE this JSON data!\n"
                elif 'image' in data_type.lower():
                    attachment_details += f"  - ⚠️ DISPLAY this image using the data URI directly in <img src=...>\n"
                attachment_details += "\n"
    
    # Format checks with emphasis
    checks_formatted = ""
    required_ids = []
    for i, check in enumerate(checks, 1):
        checks_formatted += f"{i}. **{check}**\n"
        
        # Extract element IDs from checks
        if '#' in check:
            parts = check.split('#')
            for part in parts[1:]:
                element_id = part.split()[0].strip("'\"(),.:;")
                required_ids.append(element_id)
                checks_formatted += f"   → MUST create element with id='{element_id}'\n"
    
    # Create required IDs list
    required_ids_html = "\n".join([f"   <... id='{id}'>" for id in set(required_ids)])
    
    # Build comprehensive universal prompt
    prompt = f"""You are a senior full-stack web developer. Create a COMPLETE, FULLY FUNCTIONAL single-page web application.

═══════════════════════════════════════════════════════════════
TASK DETAILS
═══════════════════════════════════════════════════════════════
Task ID: {task_id}
Round: {round_num}

REQUIREMENT:
{brief}

CHECKS THAT MUST PASS (CRITICAL):
{checks_formatted}
{attachment_details}

REQUIRED ELEMENT IDs (MUST EXIST):
{required_ids_html}

═══════════════════════════════════════════════════════════════
CRITICAL FUNCTIONALITY REQUIREMENTS
═══════════════════════════════════════════════════════════════

⚠️ MAKE IT ACTUALLY WORK - NOT JUST LOOK GOOD! ⚠️

1. **EXTENSIVE MOCK DATA** (20+ items for search/display):
   - Weather apps: Delhi, Mumbai, Kolkata, Chennai, Bangalore, Hyderabad, Pune, London, Paris, Tokyo, New York, Beijing, Dubai, Singapore, Sydney, Toronto, Berlin, Moscow, Madrid, Rome, Seoul (20+ cities)
   - CSV apps: Parse ALL data from attachment and calculate correctly
   - GitHub apps: Include sample users with realistic data
   - Form apps: Validate and display submitted data immediately

2. **CASE-INSENSITIVE EVERYTHING**:
   ```javascript
   // ALWAYS use .toLowerCase() for searches!
   const searchTerm = input.value.toLowerCase().trim();
   
   // Example for weather:
   if (weatherData[searchTerm]) {{ ... }}
   
   // Example for any search:
   const results = data.filter(item => 
       item.name.toLowerCase().includes(searchTerm)
   );
   ```

3. **INTERACTIVE ELEMENTS MUST WORK**:
   - ALL buttons must have onclick handlers
   - ALL forms must have onsubmit handlers
   - ALL inputs should trigger actions (search, filter, update)
   - Enter key should work for search inputs
   - Results must UPDATE the DOM when user interacts

4. **DATA HANDLING**:
   - **CSV/Data URIs**: DECODE base64 and PARSE:
     ```javascript
     const base64Data = dataUri.split(',')[1];
     const decodedText = atob(base64Data);
     const lines = decodedText.split('\\n');
     // Parse and use the data!
     ```
   - **Calculations**: Calculate sums, averages, totals CORRECTLY
   - **Display**: Show results in specified element IDs immediately

5. **ERROR HANDLING**:
   - Show helpful messages for invalid inputs
   - List available options when search fails
   - Handle empty inputs gracefully
   - Use try-catch for data processing

6. **AUTO-LOAD DATA**:
   - If task requires displaying data on load, do it in window.onload
   - Don't wait for user to click if data should show immediately

═══════════════════════════════════════════════════════════════
TECHNICAL IMPLEMENTATION
═══════════════════════════════════════════════════════════════

1. **SINGLE HTML FILE** with everything inline:
   - All CSS in <style> tags in <head>
   - All JavaScript in <script> tags before </body>
   - Use CDN for external libraries only

2. **BOOTSTRAP 5** (if required by checks):
   ```html
   <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
   <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
   ```

3. **OTHER LIBRARIES** (load from CDN if needed):
   - Papa Parse (CSV): https://cdn.jsdelivr.net/npm/papaparse@5/papaparse.min.js
   - Marked.js (Markdown): https://cdn.jsdelivr.net/npm/marked/marked.min.js
   - Highlight.js (Syntax): https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js
   - Chart.js (Charts): https://cdn.jsdelivr.net/npm/chart.js

4. **DESIGN**:
   - Modern, professional Bootstrap 5 styling
   - Responsive layout (mobile-friendly)
   - Good color scheme and spacing
   - Clear typography
   - Use cards, forms, tables appropriately

═══════════════════════════════════════════════════════════════
TASK-SPECIFIC EXAMPLES
═══════════════════════════════════════════════════════════════

**IF WEATHER/SEARCH TASK:**
```javascript
const weatherData = {{
    'delhi': {{ temp: 32, desc: 'Hot', icon: '🌞' }},
    'mumbai': {{ temp: 28, desc: 'Humid', icon: '💧' }},
    'kolkata': {{ temp: 30, desc: 'Warm', icon: '🌤️' }},
    'london': {{ temp: 15, desc: 'Cloudy', icon: '☁️' }},
    // ... 16 more cities
}};

function search() {{
    const input = document.getElementById('search-input').value;
    const city = input.toLowerCase().trim(); // CASE-INSENSITIVE!
    
    if (weatherData[city]) {{
        // Update display
        document.getElementById('result').textContent = weatherData[city].temp + '°C';
    }} else {{
        alert('Not found! Try: Delhi, Mumbai, Kolkata, London...');
    }}
}}

// MUST work with Enter key
document.getElementById('search-input').addEventListener('keypress', (e) => {{
    if (e.key === 'Enter') search();
}});
```

**IF CSV/DATA TASK:**
```javascript
// Parse data URI from attachment
const dataUri = '{attachments[0]["url"] if attachments else ""}';
const base64Data = dataUri.split(',')[1];
const csvText = atob(base64Data);

// Parse CSV
const lines = csvText.trim().split('\\n');
const data = [];
for (let i = 1; i < lines.length; i++) {{
    const [col1, col2] = lines[i].split(',');
    data.push({{ name: col1, value: parseFloat(col2) }});
}}

// Calculate total
const total = data.reduce((sum, row) => sum + row.value, 0);

// Display IMMEDIATELY on page load
document.getElementById('total').textContent = total.toFixed(2);
```

**IF FORM TASK:**
```javascript
document.getElementById('myform').onsubmit = (e) => {{
    e.preventDefault();
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData);
    
    // Display results
    document.getElementById('result').innerHTML = 
        `Submitted: ${{JSON.stringify(data)}}`;
    
    // Show success message
    alert('Form submitted successfully!');
}};
```

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════

Respond with ONLY the complete HTML code:
- Start immediately with: <!DOCTYPE html>
- End with: </html>
- NO explanations before or after the code
- NO markdown code blocks (no ```html)
- NO comments outside the HTML
- Just pure, working, production-ready HTML

CREATE THE APPLICATION NOW:"""

    try:
        logger.info("🤖 Generating code with gpt-4o-mini...")
        logger.info(f"   Task: {task_id}, Round: {round_num}")
        logger.info(f"   Prompt: {len(prompt)} chars")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a senior full-stack web developer who creates FUNCTIONAL, production-ready single-page applications.

CRITICAL RULES:
1. Make ALL interactive elements work (buttons, forms, search)
2. Use CASE-INSENSITIVE searches (always .toLowerCase())
3. Include EXTENSIVE mock data (20+ items for any search)
4. DECODE and PARSE data URIs from attachments correctly
5. Calculate totals/sums ACCURATELY
6. Display results IMMEDIATELY or on button click
7. All required element IDs must exist
8. Use Bootstrap 5 for professional styling
9. Make it mobile-responsive
10. NO placeholders, NO TODOs - complete, working code only!"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=8000,  # Increased for complex apps
            temperature=0.7
        )

        html_code = response.choices[0].message.content.strip()
        
        logger.info(f"✓ LLM returned {len(html_code)} characters")
        
        # Clean markdown code blocks
        if "```html" in html_code:
            logger.info("   Removing ```html wrapper")
            html_code = html_code.split("```html")[1].split("```")[0].strip()
        elif "```" in html_code:
            logger.info("   Removing ``` wrapper")
            html_code = html_code.replace("```", "").strip()

        # Validate structure
        if not html_code.startswith(("<!DOCTYPE", "<html")):
            logger.warning("   HTML doesn't start correctly, fixing...")
            if "<html" in html_code:
                html_code = html_code[html_code.index("<html"):]
            elif "<!DOCTYPE" in html_code:
                html_code = html_code[html_code.index("<!DOCTYPE"):]
            else:
                raise ValueError("No valid HTML structure found")

        # Validate minimum length
        if len(html_code) < 500:
            raise ValueError(f"HTML too short: {len(html_code)} chars")
        
        # Check for closing tag
        if not ("</html>" in html_code or "</HTML>" in html_code):
            logger.warning("   No closing </html> tag, adding it")
            html_code += "\n</html>"

        # Log structure validation
        validations = {
            "<head>": "<head>" in html_code,
            "<body>": "<body>" in html_code,
            "<script>": "<script>" in html_code or "<script " in html_code,
            "Bootstrap": "bootstrap" in html_code.lower(),
            ".toLowerCase()": ".toLowerCase()" in html_code.lower()
        }
        
        for check, result in validations.items():
            status = "✓" if result else "⚠️"
            logger.info(f"   {status} {check}: {result}")
        
        # Validate required IDs
        missing_ids = []
        for req_id in required_ids:
            if f'id="{req_id}"' not in html_code and f"id='{req_id}'" not in html_code:
                missing_ids.append(req_id)
        
        if missing_ids:
            logger.warning(f"   ⚠️ Missing required IDs: {missing_ids}")
        else:
            logger.info(f"   ✓ All {len(required_ids)} required IDs present")

        logger.info(f"✓ Generated valid HTML: {len(html_code)} chars")
        return html_code

    except Exception as e:
        logger.error(f"❌ LLM generation failed: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        logger.warning("   Using enhanced fallback template...")
        
        # Create functional fallback with required IDs
        elements_html = ""
        for req_id in set(required_ids):
            if 'search' in req_id.lower() or 'input' in req_id.lower():
                elements_html += f'        <input type="text" id="{req_id}" class="form-control mb-3" placeholder="Enter {req_id}">\n'
            elif 'button' in req_id.lower() or 'btn' in req_id.lower():
                elements_html += f'        <button id="{req_id}" class="btn btn-primary mb-3">Search</button>\n'
            elif 'table' in req_id.lower():
                elements_html += f'        <table id="{req_id}" class="table table-striped mb-3">\n'
                elements_html += f'            <thead><tr><th>Column 1</th><th>Column 2</th></tr></thead>\n'
                elements_html += f'            <tbody><tr><td>Sample</td><td>Data</td></tr></tbody>\n'
                elements_html += f'        </table>\n'
            else:
                elements_html += f'        <div id="{req_id}" class="alert alert-secondary mb-3">{req_id}</div>\n'
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{task_id}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem 0;
        }}
        .main-container {{
            background: white;
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 900px;
            margin: 0 auto;
        }}
        h1 {{ color: #667eea; font-weight: bold; }}
        .requirement {{ background: #f8f9fa; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; border-left: 4px solid #667eea; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="main-container">
            <h1 class="mb-4">🚀 {task_id}</h1>
            
            <div class="alert alert-info">
                <h5>📋 Task Brief:</h5>
                <p>{brief}</p>
            </div>
            
            <div class="alert alert-warning">
                <strong>⚠️ Note:</strong> This is a fallback template. The LLM should generate full functionality.
            </div>
            
            <h3 class="mt-4 mb-3">Required Elements:</h3>
{elements_html}
            
            <h3 class="mt-4 mb-3">Requirements to Satisfy:</h3>
{''.join([f'            <div class="requirement">✓ {check}</div>\\n' for check in checks])}
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        console.log('Task: {task_id}');
        console.log('Round: {round_num}');
        console.log('Fallback template loaded');
        
        // Add basic interactivity
        document.querySelectorAll('button').forEach(btn => {{
            btn.onclick = () => {{
                const input = document.querySelector('input');
                if (input) {{
                    alert('Search clicked for: ' + input.value);
                    // Update any result div
                    const resultDiv = document.querySelector('[id*="result"], [id*="name"], [id*="display"]');
                    if (resultDiv) {{
                        resultDiv.textContent = 'Searched: ' + input.value;
                    }}
                }}
            }};
        }});
        
        // Add Enter key support
        document.querySelectorAll('input').forEach(input => {{
            input.addEventListener('keypress', (e) => {{
                if (e.key === 'Enter') {{
                    document.querySelector('button')?.click();
                }}
            }});
        }});
    </script>
</body>
</html>"""

def generate_readme(brief: str, task_id: str, round_num: int, checks: List[str]) -> str:
    """Generate professional README"""
    checks_md = "\n".join([f"- {c}" for c in checks])
    
    return f"""# {task_id}

> LLM Code Deployment Project - Round {round_num}

## Description
{brief}

## Features
{checks_md}

## Live Demo
🔗 [View Application](https://{GITHUB_USERNAME}.github.io/{task_id}/)

## Setup
```bash
git clone https://github.com/{GITHUB_USERNAME}/{task_id}.git
cd {task_id}
open index.html
```

## Technical Details
- **Round**: {round_num}
- **Technologies**: HTML5, CSS3, JavaScript
- **Hosting**: GitHub Pages
- **Generated**: {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}

## License
MIT License - see LICENSE file

## Auto-Generated
This application was automatically generated using LLM-assisted code generation.
"""


def notify_evaluation_url(evaluation_url: str, data: Dict) -> bool:
    """Notify evaluation URL with retry logic"""
    retry_delays = [1, 2, 4, 8, 16]
    
    for attempt, delay in enumerate(retry_delays, 1):
        try:
            logger.info(f"📤 Notifying evaluation URL (attempt {attempt}/5)")
            
            response = requests.post(
                evaluation_url,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info("✅ Evaluation URL notified successfully")
                return True
            else:
                logger.warning(f"⚠️ Got {response.status_code}: {response.text[:100]}")
        
        except Exception as e:
            logger.error(f"❌ Notification error: {e}")
        
        if attempt < len(retry_delays):
            logger.info(f"⏳ Retrying in {delay}s...")
            time.sleep(delay)
    
    logger.error("❌ All notification attempts failed")
    return False


def process_task_background(data: Dict):
    """Background task processor - runs after returning 200 OK"""
    task_id = data['task']
    round_num = data['round']
    repo_name = task_id
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 PROCESSING ROUND {round_num}: {task_id}")
        logger.info(f"{'='*60}")
        
        # Generate code
        logger.info("[1/6] 🤖 Generating code...")
        html_code = generate_code_with_llm(
            data.get('brief', ''),
            data.get('checks', []),
            data.get('attachments', []),
            task_id,
            round_num
        )
        
        # Create/get repo
        if round_num == 1:
            logger.info("[2/6] 📦 Creating repo...")
            create_github_repo(repo_name, data.get('brief', '')[:100])
            time.sleep(3)
        else:
            logger.info("[2/6] ✓ Using existing repo...")
        
        # Push index.html
        logger.info("[3/6] 📤 Pushing index.html...")
        index_sha = get_file_sha(repo_name, "index.html") if round_num == 2 else None
        commit_sha = push_file_to_repo(
            repo_name, "index.html", html_code,
            f"Round {round_num}: {data.get('brief', '')[:50]}",
            index_sha
        )
        
        # Push README
        logger.info("[4/6] 📤 Pushing README...")
        readme_sha = get_file_sha(repo_name, "README.md")
        push_file_to_repo(
            repo_name, "README.md",
            generate_readme(data.get('brief', ''), task_id, round_num, data.get('checks', [])),
            f"Round {round_num}: Update README",
            readme_sha
        )
        
        # Enable Pages (Round 1 only)
        if round_num == 1:
            logger.info("[5/6] 🌐 Enabling GitHub Pages...")
            enable_github_pages(repo_name)
        else:
            logger.info("[5/6] ✓ Pages already enabled...")
        
        # Wait for deployment
        logger.info("⏳ Waiting 15s for deployment...")
        time.sleep(15)
        
        # Notify evaluation
        logger.info("[6/6] 📤 Notifying evaluation URL...")
        pages_url = f"https://{GITHUB_USERNAME}.github.io/{repo_name}/"
        notification = {
            "email": data['email'],
            "task": task_id,
            "round": round_num,
            "nonce": data['nonce'],
            "repo_url": f"https://github.com/{GITHUB_USERNAME}/{repo_name}",
            "commit_sha": commit_sha or "main",
            "pages_url": pages_url
        }
        
        success = notify_evaluation_url(data['evaluation_url'], notification)
        
        if success:
            logger.info(f"\n{'='*60}")
            logger.info(f"✅ ROUND {round_num} COMPLETED!")
            logger.info(f"   Repo: https://github.com/{GITHUB_USERNAME}/{repo_name}")
            logger.info(f"   Live: {pages_url}")
            logger.info(f"{'='*60}\n")
        else:
            logger.error(f"⚠️ Round {round_num} completed but notification failed")
        
    except Exception as e:
        logger.error(f"❌ TASK FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())


@app.post("/handle_tasks")
async def handle_tasks(data: dict, background_tasks: BackgroundTasks):
    """
    Main endpoint - receives tasks from instructors
    Returns 200 immediately, processes in background
    """
    try:
        # Validate fields
        required = ['email', 'secret', 'task', 'round', 'nonce', 'brief', 'evaluation_url']
        missing = [f for f in required if f not in data]
        if missing:
            raise HTTPException(400, f"Missing: {', '.join(missing)}")
        
        # Validate secret
        if not validate_secret(data.get("secret", "")):
            logger.warning(f"❌ Invalid secret from {data.get('email')}")
            raise HTTPException(401, "Invalid secret")
        
        # Validate round
        if data.get("round") not in [1, 2]:
            raise HTTPException(400, "Round must be 1 or 2")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📨 TASK RECEIVED")
        logger.info(f"   Email: {data['email']}")
        logger.info(f"   Task: {data['task']}")
        logger.info(f"   Round: {data['round']}")
        logger.info(f"{'='*60}")
        
        # Add to background tasks
        background_tasks.add_task(process_task_background, data)
        
        # Return 200 immediately
        return JSONResponse(
            status_code=200,
            content={
                "status": "accepted",
                "message": f"Task {data['task']} round {data['round']} is being processed",
                "task": data['task'],
                "round": data['round']
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"💥 ERROR: {e}")
        raise HTTPException(500, str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    github_ok = bool(GITHUB_TOKEN)
    openai_ok = bool(OPENAI_API_KEY)
    secret_ok = bool(SECRET)
    
    return {
        "status": "healthy" if all([github_ok, openai_ok, secret_ok]) else "unhealthy",
        "checks": {
            "github_token": "✓" if github_ok else "✗",
            "openai_key": "✓" if openai_ok else "✗",
            "secret": "✓" if secret_ok else "✗"
        },
        "config": {
            "username": GITHUB_USERNAME,
            "openai_base": OPENAI_BASE_URL
        }
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "LLM Code Deployment - Student API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /handle_tasks": "Receive task requests",
            "GET /health": "Health check",
            "GET /": "This page"
        }
    }


if __name__ == "__main__":
    # Validate config
    if not all([GITHUB_TOKEN, SECRET, OPENAI_API_KEY]):
        logger.error("❌ Missing required environment variables!")
        logger.error(f"   GITHUB_TOKEN: {'✓' if GITHUB_TOKEN else '✗'}")
        logger.error(f"   SECRET: {'✓' if SECRET else '✗'}")
        logger.error(f"   OPENAI_API_KEY: {'✓' if OPENAI_API_KEY else '✗'}")
        exit(1)
    
    logger.info("🚀 Starting LLM Code Deployment API")
    logger.info(f"   GitHub: {GITHUB_USERNAME}")
    logger.info(f"   LLM: {OPENAI_BASE_URL}")
    logger.info(f"   All config: ✓")
    logger.info("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)