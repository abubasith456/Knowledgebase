# Knowledge Base Backend API

A modern FastAPI-based backend for a knowledge base system with document processing, vector search, and REST API endpoints.

## Features

- **Document Processing**: Upload and process PDF, DOCX, DOC, TXT, and MD files
- **Vector Search**: Semantic search using Jina embeddings and ChromaDB
- **Document Management**: List, delete, and manage uploaded documents
- **REST API**: Complete REST API with automatic documentation
- **CORS Support**: Cross-origin resource sharing enabled
- **Error Handling**: Comprehensive error handling and validation

## Technology Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **ChromaDB**: Vector database for storing embeddings
- **Jina Embeddings**: High-quality text embeddings
- **Docling**: Document parsing and conversion
- **Tiktoken**: Token counting and text processing
- **Pydantic**: Data validation and serialization

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd backend
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The backend uses the following default configuration:

- **Embedding Model**: `jinaai/jina-embeddings-v3`
- **Max Tokens**: 8,000
- **Chunk Size**: 1,000 tokens
- **Chunk Overlap**: 200 tokens
- **Vector DB Path**: `./chroma_db`
- **Port**: 8000

You can modify these settings in the `CompleteKBService` class initialization.

## Running the Application

### Development Mode

```bash
python main.py
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## API Endpoints

### Document Management

#### Upload Document
```http
POST /api/upload
Content-Type: multipart/form-data

file: [document file]
```

**Response**:
```json
{
  "success": true,
  "file_name": "document.pdf",
  "token_count": 1500,
  "chunks_created": 3,
  "chunks_deleted": 0,
  "needs_chunking": true,
  "action": "added",
  "message": "Successfully added 'document.pdf' with 3 chunks"
}
```

#### List Documents
```http
GET /api/documents
```

**Response**:
```json
{
  "success": true,
  "documents": [
    {
      "filename": "document.pdf",
      "chunks": 3,
      "file_type": "pdf",
      "upload_timestamp": "2024-01-01T12:00:00",
      "document_hash": "abc123...",
      "document_id": "uuid-here",
      "file_size": 1024000
    }
  ],
  "total_count": 1
}
```

#### Delete Documents
```http
DELETE /api/documents
Content-Type: application/json

{
  "document_ids": ["uuid1", "uuid2"]
}
```

**Response**:
```json
{
  "success": true,
  "deleted_documents": 2,
  "deleted_chunks": 6,
  "document_names": ["doc1.pdf", "doc2.docx"],
  "message": "Deleted 2 documents with 6 chunks"
}
```

### Search

#### Query Knowledge Base
```http
POST /api/query
Content-Type: application/json

{
  "query": "What is machine learning?",
  "n_results": 5,
  "document_ids": ["uuid1", "uuid2"]  // Optional: filter by specific documents
}
```

**Response**:
```json
{
  "success": true,
  "query": "What is machine learning?",
  "results": {
    "documents": [["chunk1", "chunk2"]],
    "metadatas": [[{"source": "doc1.pdf", "chunk_index": "0"}]],
    "distances": [[0.1, 0.2]]
  },
  "count": 2,
  "filtered_by_documents": ["uuid1", "uuid2"]
}
```

### System Information

#### Get Statistics
```http
GET /api/stats
```

**Response**:
```json
{
  "total_chunks": 15,
  "unique_documents": 3,
  "documents": ["doc1.pdf", "doc2.docx", "doc3.txt"],
  "collection_name": "knowledge_base"
}
```

#### Health Check
```http
GET /api/health
```

**Response**:
```json
{
  "status": "healthy",
  "service": "Knowledge Base API"
}
```

## File Format Support

The backend supports the following file formats:

- **PDF** (.pdf)
- **Microsoft Word** (.docx, .doc)
- **Plain Text** (.txt)
- **Markdown** (.md)

## Document Processing Pipeline

1. **File Upload**: Accept file via multipart form data
2. **Document Parsing**: Use Docling to extract text content
3. **Content Analysis**: Calculate tokens and determine chunking needs
4. **Text Chunking**: Split content into manageable chunks with overlap
5. **Embedding Generation**: Create vector embeddings using Jina model
6. **Vector Storage**: Store embeddings and metadata in ChromaDB
7. **Response**: Return processing results and statistics

## Error Handling

The API includes comprehensive error handling:

- **File Validation**: Check file type and size
- **Processing Errors**: Handle document parsing failures
- **Database Errors**: Manage ChromaDB connection issues
- **HTTP Status Codes**: Proper status codes for different error types

## Development

### Project Structure

```
backend/
├── main.py              # Main FastAPI application
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Adding New Features

1. **New Endpoints**: Add routes to the FastAPI app
2. **Data Models**: Create Pydantic models for request/response validation
3. **Service Methods**: Extend the `CompleteKBService` class
4. **Error Handling**: Add appropriate exception handling

### Testing

```bash
# Run with uvicorn for development
uvicorn main:app --reload

# Test endpoints using curl
curl -X POST "http://localhost:8000/api/upload" -F "file=@test.pdf"
curl -X POST "http://localhost:8000/api/query" -H "Content-Type: application/json" -d '{"query": "test query"}'
```

## Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

- `API_HOST`: Server host (default: 0.0.0.0)
- `API_PORT`: Server port (default: 8000)
- `VECTOR_DB_PATH`: ChromaDB storage path (default: ./chroma_db)
- `EMBEDDING_MODEL`: Embedding model name (default: jinaai/jina-embeddings-v3)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.