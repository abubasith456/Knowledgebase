# Knowledge Base System

A modern, full-stack knowledge base system with AI-powered document processing and semantic search capabilities. This project has been separated into a clean backend API and a modern React frontend.

## 🚀 Features

- **Document Processing**: Upload and process PDF, DOCX, DOC, TXT, and MD files
- **AI-Powered Search**: Semantic search using Jina embeddings and vector similarity
- **Smart Indexing**: Auto and Manual indexing modes with intelligent token detection
- **Multiple Embedding Models**: Support for Jina Embeddings v3 and Qwen3 0.6B
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

## 🧠 Indexing Modes

### Auto Mode (Default)
- **Smart Token Detection**: Automatically detects if content needs chunking
- **Token Threshold**: 7,000 tokens (configurable)
- **Behavior**: 
  - If content < threshold: Store as raw text (no embedding)
  - If content ≥ threshold: Chunk and embed
- **Benefits**: Optimizes storage and processing for small documents

### Manual Mode
- **Character-based Chunking**: Manual control over chunk size and overlap
- **Configurable Parameters**:
  - Chunk Size: 1,000 characters (default)
  - Chunk Overlap: 200 characters (default)
- **Benefits**: Precise control for specific use cases

## 🤖 Embedding Models

### Supported Models

1. **Jina Embeddings v3** (Default)
   - High-quality embeddings
   - 8K context window
   - Optimized for semantic search

2. **Qwen3 0.6B**
   - Lightweight model
   - Fast processing
   - Good for resource-constrained environments

### Model Selection
- Choose during document upload
- Different models can be used for different documents
- Automatic fallback to default model if loading fails

## 📖 Usage

### 1. Upload Documents

1. Navigate to the "Upload Documents" tab
2. Configure indexing settings (optional):
   - **Indexing Mode**: Auto or Manual
   - **Embedding Model**: Jina v3 or Qwen3 0.6B
   - **Token Threshold**: For auto mode (1,000-10,000)
   - **Chunk Size/Overlap**: For manual mode
3. Drag and drop files or click to select
4. Supported formats: PDF, DOCX, DOC, TXT, MD
5. View upload progress and results

### 2. Search Knowledge Base

1. Go to the "Search Knowledge Base" tab
2. Enter your query in natural language
3. Adjust number of results if needed
4. Optionally filter by specific documents
5. View search results with similarity scores

### 3. Manage Documents

1. Visit the "Manage Documents" tab
2. View all uploaded documents and statistics
3. See indexing mode and embedding model for each document
4. View embedded vs raw chunk counts
5. Select documents for bulk operations
6. Delete documents as needed

## 🔌 API Endpoints

### Document Management

- `POST /api/upload` - Upload and process documents
  - Query params: `indexing_mode`, `embedding_model`, `manual_chunk_size`, `manual_chunk_overlap`, `auto_token_threshold`
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

- **FileUpload**: Drag-and-drop interface with advanced indexing options
- **QueryInterface**: Advanced search with document filtering
- **DocumentManager**: Comprehensive document management with indexing info
- **Toast Notifications**: User-friendly feedback
- **Responsive Design**: Mobile-first approach

### Indexing Configuration

- **Auto Mode**: Smart token detection with configurable threshold
- **Manual Mode**: Character-based chunking with overlap
- **Model Selection**: Choose between Jina v3 and Qwen3 0.6B
- **Real-time Preview**: See current configuration before upload

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
- **Indexing**: Added smart indexing modes and model selection

### Benefits

- **Better Performance**: Separated concerns and optimized loading
- **Modern UI**: Professional, responsive design
- **Scalability**: Independent scaling of frontend and backend
- **Maintainability**: Clear separation of concerns
- **Developer Experience**: Better debugging and development tools
- **Flexibility**: Multiple indexing modes and embedding models

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
- **Qwen**: Alternative embedding model

## 📞 Support

For questions and support:
- Check the documentation in `backend/README.md` and `frontend/README.md`
- Visit the API documentation at http://localhost:8000/docs
- Open an issue on GitHub