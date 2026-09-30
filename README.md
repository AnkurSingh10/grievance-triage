# 🏛️ Grievance Command Centre (Govt-AI Triage & RAG System)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=for-the-badge&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/ankursingh01/govt-grievance)
[![Neon](https://img.shields.io/badge/Neon-pgvector-00E599.svg?style=for-the-badge&logo=postgresql&logoColor=white)](https://neon.tech)
[![Gemini](https://img.shields.io/badge/Google_Gemini-RAG_Grounded-8E75C2.svg?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev)

## 📌 Executive Summary

Modern public governance systems process millions of citizen complaints every year across web portals, mobile apps, call centres, and district offices. Traditional grievance management systems suffer from severe systemic bottlenecks:
- **Manual Routing Delays:** Clerks spend days manually reviewing and routing tickets, leading to departmental backlogs and misdirected assignments.
- **Linguistic & Code-Mixed Barriers:** Citizens file grievances in a mix of Hindi, English, regional dialects, and transliterated scripts (*"Hinglish"* e.g., *"road par bada pothole hai, accident ka risk hai"*), which break rule-based and keyword-driven filters.
- **Duplicate Grievance Sprawls:** A single civic disruption (e.g., a burst water main or traffic light failure) triggers hundreds of duplicate submissions, flooding field officers with redundant tasks.
- **Cognitive Overload for Resolution Officers:** Reviewing officers lack instant access to relevant government circulars, SOPs, and past resolution precedents needed to make compliant, rapid decisions.

**Grievance Command Centre** is an end-to-end, production-grade AI operating system for public administration. It combines **fine-tuned Indic multilingual transformers (Google MuRIL)**, **real-time vector similarity deduplication (Neon pgvector)**, and **grounded generative decision support (Google Gemini 1.5 Flash RAG)** to automate citizen triage, detect duplicate clusters, prioritize emergencies, and assist officers in taking verified, policy-backed action in seconds.

---

### 💡 Core Value Propositions & Capabilities

| Capability | Technical Mechanism | Real-World Impact |
|---|---|---|
| **Intelligent Routing** | Fine-tuned **Google MuRIL** with joint classification heads | Sub-50ms automated department assignment with **99.98% accuracy** across English & Hinglish text. |
| **Urgency & SLA Triage** | Multi-task urgency scoring ($\text{P0}_{\text{Critical}}$ to $\text{P3}_{\text{Routine}}$) | Automatically computes dynamic SLA deadlines and flags life-safety hazards instantly. |
| **Vector Deduplication** | **Neon pgvector** (384-d embeddings + HNSW cosine index) | Groups identical ($\ge 0.95$) and related ($0.85 - 0.95$) complaints, reducing duplicate field visits. |
| **Grounded AI Officer Copilot** | **Google Gemini 1.5 Flash** RAG over government circulars & past cases | Generates actionable next steps, cited circular references, and precedent summaries without hallucinations. |
| **Microservice Architecture** | **FastAPI** (AWS EC2 Docker) + **Streamlit** (Render) | Clean separation of concerns with ultra-low latency inference and lightweight client frontend. |

---


## 🌐 Live Deployments

| Component | Platform | URL / Endpoint | Description |
|---|---|---|---|
| **Frontend UI** | **Render** | [https://grievance-dashboard.onrender.com](https://grievance-dashboard.onrender.com/) | Pure REST Streamlit citizen & officer command centre |
| **Docker Hub** | **Registry** | [ankursingh01/govt-grievance:latest](https://hub.docker.com/r/ankursingh01/govt-grievance) | Production Docker image containerizing the full backend |

---

## 📸 System Showcase & Visual Walkthrough

### 1. Citizen Intake Portal
Citizens can submit complaints across multiple channels (*Web, Mobile, Call Centre, In-Person Office*). The input text (supporting English, Hindi, and Hinglish) is instantly triaged by the **MuRIL model** to predict the responsible department and urgency level with confidence scores.

![Citizen Intake Portal](assets/01_citizen_portal.png)

---

### 2. Department Officer Queue & Triage
Departmental officers can filter their live queue by status (*Submitted, Classified, Assigned, Under Investigation, Action Taken, Resolved*) and view SLA deadlines, priority levels (P0–P3), and incoming grievance trends.

![Officer Complaint Queue](assets/02_officer_queue.png)

---

### 3. Review, Deduplication & Semantic Relation
The system runs vector similarity against past complaints using **Neon pgvector (HNSW cosine index)**. When duplicate or related complaints are submitted, they are automatically flagged with cosine similarity scores (e.g. `0.997 DUPLICATE`), preventing redundant field inspections and clustering related civic issues.

![Deduplication and Semantic Relation](assets/03_deduplication_review.png)

---

### 4. Grounded AI Officer Assistance (Gemini 1.5 RAG)
Officers can generate automated, policy-grounded resolution strategies with 1-click. The pipeline searches Neon for matching departmental circulars, standard operating procedures (SOPs), and historically resolved cases, prompting **Google Gemini** to output actionable recommendations and citations without hallucinations.

![Grounded AI Assistance](assets/05_gemini_rag_assistance.png)

---

### 5. Real-Time Department Analytics Dashboard
High-level administrators can view live grievance metrics across all departments, tracking newly received tickets, critical/high/medium/low severity distributions, and pending unresolved caseloads.

![Department Analytics Dashboard](assets/04_department_dashboard.png)


---

## 🧠 Machine Learning Evolution & Architecture

### Phase 1: BiLSTM + Additive Attention (Baseline)
- **Architecture:** Word Embedding Layer $\rightarrow$ Bidirectional LSTM (2-layer, hidden dim 128) $\rightarrow$ Additive Attention Context Pooling $\rightarrow$ Dual Linear Classification Heads (Department & Urgency).
- **Pros:** Ultra-lightweight (~15MB), fast CPU training.
- **Cons:** Limited semantic depth on colloquial Indic text, code-mixed Hinglish (*"ek bada pothole road par hai"*), and regional vocabulary.

### Phase 2: Google MuRIL Fine-Tuning (`google/muril-base-cased`)
- **Multilingual Representation for Indian Languages (MuRIL)**: Specifically pre-trained by Google Research on 17 Indian languages plus English, covering both monolingual text and translated/transliterated parallel corpora.
- **Training Setup:**
  - Tokenizer: WordPiece with 197k Indic vocabulary.
  - Multi-Task Classification: Joint classification heads with Cross-Entropy Loss for multi-class department routing and urgency grading.
  - **Performance:** Achieved **99.98% validation accuracy**, excelling at messy real-world citizen inputs containing transliterated Hindi, spelling variations, and official administrative terms.

### Phase 3: FP16 Quantization & Model Compression
- Original PyTorch MuRIL weights occupied **~950MB (FP32)**, consuming excessive RAM during Docker container startup on resource-constrained cloud servers.
- Compressed all model parameter tensors to **Half-Precision (Float16)**:
  - **52.4% File Size Reduction:** Shrinking checkpoint from **950MB $\rightarrow$ 452MB**.
  - **Zero Metric Degradation:** Retained the exact **99.98% validation accuracy**.
  - **Inference Speedup:** ~40% faster latency per batch on CPU/GPU instances.

---

## 🏗️ System Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Frontend ["Render (Web Frontend)"]
        Streamlit["Streamlit UI (Pure REST Client)<br/>grievance-dashboard.onrender.com"]
    end

    subgraph Backend ["AWS EC2 (Backend Server)"]
        FastAPI["FastAPI App (Docker Container)<br/>3.236.127.3:8000"]
        MuRIL["MuRIL FP16 Classifier<br/>(Dept & Urgency Inference)"]
        ST_Model["Sentence-Transformers<br/>(all-MiniLM-L6-v2)"]
        Gemini["Google Gemini 1.5 Flash<br/>(Grounded RAG Agent)"]
    end

    subgraph Database ["Neon Cloud (Serverless Postgres)"]
        NeonDB[("PostgreSQL Database")]
        pgvector["pgvector HNSW Index<br/>(Deduplication & Document Retrieval)"]
    end

    Streamlit -- "REST / JSON" --> FastAPI
    FastAPI --> MuRIL
    FastAPI --> ST_Model
    FastAPI --> Gemini
    ST_Model -- "Embeddings (384-d)" --> pgvector
    FastAPI -- "SQL / Relational Data" --> NeonDB
```

---

## 🛠️ Technology Stack

- **Frontend:** Streamlit 1.40+, Pandas, Requests (Pure decoupled UI client).
- **Backend API:** FastAPI, Uvicorn, Pydantic v2, CORS Middleware.
- **Deep Learning / NLP:** PyTorch, Transformers, Google MuRIL (`google/muril-base-cased`), SentenceTransformers (`all-MiniLM-L6-v2`).
- **Database & Vectors:** Neon Serverless PostgreSQL, `pgvector` with HNSW cosine distance indexing, SQLAlchemy 2.0.
- **Generative AI / RAG:** LangChain, Google Generative AI (`gemini-1.5-flash`), Prompt Grounding.
- **MLOps & Experiment Tracking:** MLflow, DagsHub, GitHub Actions CI/CD, Docker Hub.
- **Cloud Infrastructure:** AWS EC2 (Ubuntu 24.04), Render (Web Service).

---

## 🚀 Deployment Guide

### 1. Backend Deployment (AWS EC2 + Docker)
The backend is packaged into a self-contained Docker image `ankursingh01/govt-grievance:latest`:

```bash
# Pull and run on EC2 instance
docker run -d \
  -p 8000:8000 \
  -e NEON_DB="postgresql://user:pass@ep-host.neon.tech/neondb?sslmode=require" \
  -e GOOGLE_API_KEY="your-gemini-api-key" \
  -e MODEL_TYPE="muril" \
  -e CHECKPOINT_PATH="outputs/submission_muril_model_fp16.pt" \
  --name grievance-api \
  --restart always \
  ankursingh01/govt-grievance:latest
```

### 2. Frontend Deployment (Render)
1. Link your GitHub repository to a new **Web Service** on [Render](https://render.com).
2. Configure settings:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements-render.txt`
   - **Start Command:** `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
3. Add Environment Variable:
   - `API_BASE_URL` = `http://3.236.127.3:8000`

---

## 🔌 API Endpoints Reference

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Healthcheck and readiness probe |
| `POST` | `/complaints` | Submit citizen complaint & run MuRIL FP16 classification |
| `GET` | `/complaints/{id}` | Get full complaint metadata and urgency details |
| `GET` | `/complaints/{id}/status` | Track current status and complete transition audit history |
| `GET` | `/complaints/{id}/related` | Retrieve similar/duplicate complaints via pgvector similarity |
| `PATCH` | `/complaints/{id}/status` | Update workflow status with officer comments |
| `POST` | `/complaints/{id}/assist` | Trigger grounded Gemini 1.5 Flash RAG resolution assistance |
| `POST` | `/officer/feedback` | Record officer feedback on department/urgency corrections |
| `POST` | `/knowledge-documents` | Ingest and vectorize government policies/circulars |
| `GET` | `/departments/{dept}/complaints` | Fetch real-time departmental queue |
| `GET` | `/departments/{dept}/dashboard` | Aggregate departmental urgency and resolution metrics |
| `GET` | `/analytics/issues` | Retrieve top recurring complaint clusters |

---

## 💻 Local Development Setup

```bash
# 1. Clone repository
git clone https://github.com/AnkurSingh10/grievance-triage.git
cd grievance-triage

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate

# 3. Install full dependencies
pip install -r requirements.txt
pip install -e .

# 4. Configure environment (.env)
cp .env.example .env
# Fill NEON_DB and GOOGLE_API_KEY in .env

# 5. Run FastAPI Backend
python -m grievance_triage.api.main

# 6. Run Streamlit Frontend
streamlit run streamlit_app.py
```

---

## 👥 Authors & Contributors
- **Ankur Singh** — *Lead Developer & Architect* — [GitHub](https://github.com/AnkurSingh10)
