# 🚀 Autonomous LLM-Powered Deployment System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)
![Status](https://img.shields.io/badge/Status-Live-success.svg)

**A production-style system that takes natural-language task briefs, generates full-stack apps using LLMs, deploys them automatically to GitHub Pages, verifies deployment health, and supports iterative refinement — without manual intervention.**

**🔗 Live API**: [zenitsu2121-llm-code-deployment-api.hf.space](https://zenitsu2121-llm-code-deployment-api.hf.space)

[Overview](#-what-it-does) • [Tech Stack](#-tech-stack) • [Key Features](#-key-features) • [Results](#-results) • [Quick Start](#-quick-start)

</div>

---

## 🎯 What It Does

This project automates the **entire software deployment pipeline** — from receiving a natural language brief to deploying a live web application on GitHub Pages. No manual coding, deployment, or configuration required.

**Input**: Task description in JSON format  
**Output**: Fully functional web app with documentation, live on GitHub Pages

### Real-World Impact
- **Reduced deployment time** from hours to ~3 minutes per application
- **100% automation** of repository creation, code generation, and deployment
- **Zero manual intervention** in the entire pipeline
- Successfully deployed **6 production applications** with 100% success rate

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | FastAPI, Python 3.10+, Uvicorn |
| **AI/LLM** | Google Gemini 2.5 Flash (code generation) |
| **DevOps** | GitHub API, Git, GitHub Pages, Docker |
| **Deployment** | Hugging Face Spaces |
| **Security** | Secret scanning, input validation, authentication |

---

## ✨ Key Features

### 🤖 Intelligent Code Generation
- Converts natural language specifications into production-ready HTML/CSS/JS
- Handles multi-modal inputs (text, images, CSV, JSON)
- Generates professional README and MIT License automatically

### 🔄 Automated Git Operations
- Creates GitHub repositories programmatically via API
- Manages commits, pushes, and branch operations
- Configures GitHub Pages with health checks

### 🛡️ Production-Grade Engineering
- **Secret scanning** before every commit (prevents credential leaks)
- **Concurrent processing** with semaphore-based resource management
- **Exponential backoff** retry mechanism for API calls
- **Background task execution** for non-blocking operations

### 📊 Two-Round Deployment Cycle
1. **Round 1**: Initial application generation and deployment
2. **Round 2**: Surgical updates to existing codebase based on new requirements

---

## 🏗️ System Architecture
```
Request → Authentication → LLM Generation → Secret Scan → Git Operations → GitHub Pages → Webhook Notification
```

**Key Components:**
- **FastAPI Server**: Handles incoming requests and orchestrates workflow
- **LLM Engine**: Gemini 2.5 Flash generates code based on task brief
- **GitHub Integration**: Automated repo management and Pages deployment
- **Security Layer**: Pre-commit secret detection and validation

---

## 📈 Results

| Metric | Achievement |
|--------|-------------|
| **Total Deployments** | 6 applications successfully deployed |
| **Success Rate** | 100% (6/6 tasks) |
| **Avg. Deployment Time** | ~180 seconds end-to-end |
| **GitHub Pages Accessibility** | 100% within 5 minutes |
| **Code Generation Time** | 10-25 seconds per task |

### Sample Deployments
- ✅ **Markdown to HTML Converter** (119.8s) - Syntax highlighting with highlight.js
- ✅ **GitHub User Lookup Tool** (189.3s) - API integration + account age calculation
- ✅ **Todo List Application** (241.8s) - Full CRUD with filters and statistics
- ✅ **Weather Dashboard** (535.0s) - Temperature unit conversion + responsive UI

Each deployment includes:
- Fully functional single-page application
- Professional documentation with live demo link
- MIT License with proper copyright
- GitHub repository with complete history

---

## 🚀 Quick Start

### Local Setup
```bash
# Clone repository
git clone https://github.com/yourusername/llm-code-deployment.git
cd llm-code-deployment

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Add your GEMINI_API_KEY, GITHUB_TOKEN, GITHUB_USERNAME, STUDENT_SECRET

# Run server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Usage Example
```bash
curl -X POST http://localhost:8000/ready \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "secret": "your_secret",
    "task": "calculator-app",
    "round": 1,
    "brief": "Create a calculator with memory functions",
    "checks": ["Form exists", "Display updates on click"],
    "evaluation_url": "https://webhook.site/your-id",
    "attachments": []
  }'
```

**Response**: Application deployed to `https://username.github.io/calculator-app-user/`

---

## 📡 API Documentation

### `POST /ready`
Submit a new task for automated deployment
- **Auth**: Secret-based validation
- **Response**: 200 OK with task status

### `GET /status`
Check current system status and running tasks

### `GET /health`
Health check endpoint for monitoring

### `GET /logs?lines=N`
Retrieve application logs (last N lines)

---

## 🔒 Security Features

- **Pre-commit Secret Scanning**: Detects API keys, tokens, passwords before pushing
- **Input Validation**: Pydantic models for request validation
- **Authentication**: Shared secret verification
- **Environment Isolation**: Sensitive configs stored in environment variables

---

## 🎓 Learning Outcomes

This project demonstrates:
- **Full-stack deployment automation** from scratch
- **LLM integration** for practical software engineering tasks
- **RESTful API design** with FastAPI best practices
- **Git automation** using GitHub API
- **Production deployment** on cloud platforms (Hugging Face Spaces)
- **Security-first development** with credential scanning

---

## 🔮 Future Enhancements

- [ ] PostgreSQL integration for task history and analytics
- [ ] WebSocket support for real-time status updates
- [ ] Multi-LLM support (Claude, GPT-4, Llama)
- [ ] Automated rollback on deployment failures
- [ ] CI/CD pipeline with automated testing

---

## 📞 Contact

**GitHub**: [@yourusername](https://github.com/Shubham30000)  
**Email**: 23f2005282@ds.study.iitm.ac.in  
**LinkedIn**: [linkedin.com/in/yourprofile](https://www.linkedin.com/in/shubham-singh-50a90a253/)

**Live Demo**: [Hugging Face Space](https://zenitsu2121-llm-code-deployment-api.hf.space)

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

<div align="center">

**Built with focus on automation, scalability, and production-grade engineering**

⭐ Star this repository if you find it useful!

</div>
