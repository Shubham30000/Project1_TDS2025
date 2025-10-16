# LLM Code Deployment API

Submission for IITM BS Degree - LLM Code Deployment Project

---

## 🚀 Description

This FastAPI application:

1. Receives task requests from instructors  
2. Generates web applications using LLM (GPT-4o-mini)  
3. Creates GitHub repositories automatically  
4. Deploys to GitHub Pages  
5. Notifies evaluation endpoints  

---

## 📡 API Endpoints

### POST `/handle_tasks`
Receives task requests and processes them in the background.

**Request Format:**
```json
{
  "email": "student@example.com",
  "secret": "secret_key",
  "task": "task-id",
  "round": 1,
  "nonce": "unique-nonce",
  "brief": "Task description",
  "checks": ["check1", "check2"],
  "evaluation_url": "https://example.com/evaluate",
  "attachments": []
}
