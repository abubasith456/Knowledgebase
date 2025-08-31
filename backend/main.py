# Backend FastAPI Application
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from kb_service import CompleteKBService, app as kb_app

# Load environment variables
load_dotenv()

# Create main FastAPI app
app = FastAPI(
    title="Knowledge Base API",
    description="Backend API for Knowledge Base System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the KB service routes
app.include_router(kb_app.router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Knowledge Base API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Knowledge Base API"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting Knowledge Base Backend on {host}:{port}")
    print(f"📖 API Documentation: http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port, log_level="info")
