# Knowledge Base System - Split Architecture

This project has been split into separate backend and frontend components for hosting purposes.

## Project Structure

```
├── backend/           # FastAPI backend service
│   ├── main.py       # Backend entry point
│   ├── kb_service.py  # Core knowledge base service
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
├── frontend/          # Gradio frontend application
│   ├── app.py         # Frontend application
│   ├── requirements.txt
│   └── README.md
├── docker-compose.yml # Backend deployment
└── README.md          # This file
```

## Quick Start

### 1. Start Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```
Backend will be available at: http://localhost:8000

### 2. Start Frontend
```bash
cd frontend
pip install -r requirements.txt
python app.py
```
Frontend will be available at: http://localhost:7860

## Docker Deployment

### Backend Only
```bash
docker-compose up --build
```

## Hosting Options

### Backend Hosting
- **Railway**: Deploy backend with docker-compose
- **Render**: Deploy as Python service
- **Heroku**: Deploy with Procfile
- **AWS/GCP**: Deploy to cloud platforms

### Frontend Hosting
- **Hugging Face Spaces**: Perfect for Gradio apps
- **Streamlit Cloud**: Alternative hosting for Gradio
- **Vercel/Netlify**: Deploy as static site
- **GitHub Pages**: Free hosting option
