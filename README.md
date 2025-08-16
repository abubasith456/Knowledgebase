# Knowledge Base System

A modern, full-stack knowledge base system with AI-powered document processing and semantic search capabilities. This project has been separated into a clean backend API and a modern React frontend.

## 🚀 Features

- **Document Processing**: Upload and process PDF, DOCX, DOC, TXT, and MD files
- **AI-Powered Search**: Semantic search using Jina embeddings and vector similarity
- **Modern UI**: Beautiful React frontend with Tailwind CSS
- **REST API**: Complete FastAPI backend with automatic documentation
- **Document Management**: View, manage, and delete uploaded documents
- **Real-time Updates**: Live statistics and document list updates
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## 🏗️ Architecture

This project has been separated into two distinct parts:

### Backend (FastAPI)
- **Location**: `backend/`
- **Technology**: FastAPI, ChromaDB, Jina Embeddings, Docling
- **Port**: 8000
- **Features**: Document processing, vector search, REST API

### Frontend (React)
- **Location**: `frontend/`
- **Technology**: React 18, Tailwind CSS, Axios
- **Port**: 3000
- **Features**: Modern UI, drag-and-drop uploads, search interface

## 📁 Project Structure

```
knowledge-base-system/
├── backend/                 # FastAPI Backend
│   ├── main.py             # Main FastAPI application
│   ├── requirements.txt    # Python dependencies
│   └── README.md          # Backend documentation
├── frontend/               # React Frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── api.js         # API service functions
│   │   ├── App.js         # Main application
│   │   └── index.css      # Global styles
│   ├── package.json       # Node.js dependencies
│   ├── tailwind.config.js # Tailwind configuration
│   └── README.md          # Frontend documentation
├── app.py                 # Original Gradio app (legacy)
├── main.py               # Original main file (legacy)
├── requirements.txt      # Original requirements (legacy)
└── README.md            # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+** for backend
- **Node.js 16+** for frontend
- **npm** or **yarn** for package management

### 1. Start the Backend

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

The backend will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs

### 2. Start the Frontend

```bash
# Open a new terminal and navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm start
```

The frontend will be available at:
- **Application**: http://localhost:3000

## 🔧 Configuration

### Backend Configuration

The backend uses these default settings (configurable in `backend/main.py`):

- **Embedding Model**: `jinaai/jina-embeddings-v3`
- **Max Tokens**: 8,000
- **Chunk Size**: 1,000 tokens
- **Chunk Overlap**: 200 tokens
- **Vector DB Path**: `./chroma_db`

### Frontend Configuration

Create a `.env` file in the `frontend/` directory:

```env
REACT_APP_API_URL=http://localhost:8000
```

## 📖 Usage

### 1. Upload Documents

1. Navigate to the "Upload Documents" tab
2. Drag and drop files or click to select
3. Supported formats: PDF, DOCX, DOC, TXT, MD
4. View upload progress and results

### 2. Search Knowledge Base

1. Go to the "Search Knowledge Base" tab
2. Enter your query in natural language
3. Adjust number of results if needed
4. Optionally filter by specific documents
5. View search results with similarity scores

### 3. Manage Documents

1. Visit the "Manage Documents" tab
2. View all uploaded documents and statistics
3. Select documents for bulk operations
4. Delete documents as needed

## 🔌 API Endpoints

### Document Management

- `POST /api/upload` - Upload and process documents
- `GET /api/documents` - List all documents
- `DELETE /api/documents` - Delete documents by IDs

### Search

- `POST /api/query` - Query the knowledge base
- `GET /api/stats` - Get system statistics
- `GET /api/health` - Health check

### API Documentation

Visit http://localhost:8000/docs for interactive API documentation.

## 🎨 Frontend Features

### Modern UI Components

- **FileUpload**: Drag-and-drop interface with progress indicators
- **QueryInterface**: Advanced search with document filtering
- **DocumentManager**: Comprehensive document management
- **Toast Notifications**: User-friendly feedback
- **Responsive Design**: Mobile-first approach

### Styling

- **Tailwind CSS**: Utility-first styling
- **Custom Components**: Reusable button and card styles
- **Animations**: Smooth transitions and loading states
- **Icons**: Lucide React icons throughout

## 🔄 Migration from Gradio

This project has been migrated from a single Gradio application to a separated architecture:

### What Changed

- **Backend**: Extracted FastAPI logic from Gradio app
- **Frontend**: Replaced Gradio UI with React + Tailwind CSS
- **API**: Clean REST API endpoints
- **UI/UX**: Modern, responsive interface

### Benefits

- **Better Performance**: Separated concerns and optimized loading
- **Modern UI**: Professional, responsive design
- **Scalability**: Independent scaling of frontend and backend
- **Maintainability**: Clear separation of concerns
- **Developer Experience**: Better debugging and development tools

## 🛠️ Development

### Backend Development

```bash
cd backend
uvicorn main:app --reload
```

### Frontend Development

```bash
cd frontend
npm start
```

### Testing

```bash
# Backend tests
cd backend
python -m pytest

# Frontend tests
cd frontend
npm test
```

## 🚀 Deployment

### Backend Deployment

```bash
# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Docker
docker build -t knowledge-base-backend .
docker run -p 8000:8000 knowledge-base-backend
```

### Frontend Deployment

```bash
# Build for production
cd frontend
npm run build

# Deploy build folder to any static hosting service
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Gradio**: Original UI framework
- **FastAPI**: Modern Python web framework
- **React**: Frontend framework
- **Tailwind CSS**: Utility-first CSS framework
- **ChromaDB**: Vector database
- **Jina AI**: Embedding models

## 📞 Support

For questions and support:
- Check the documentation in `backend/README.md` and `frontend/README.md`
- Visit the API documentation at http://localhost:8000/docs
- Open an issue on GitHub