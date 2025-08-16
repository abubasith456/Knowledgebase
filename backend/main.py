import os
import hashlib
import tempfile
import threading
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import uuid
import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from docling.document_converter import DocumentConverter
import chromadb
from sentence_transformers import SentenceTransformer
import tiktoken

# ===========================
# PYDANTIC MODELS FOR API
# ===========================

class QueryRequest(BaseModel):
    query: str
    n_results: int = 5
    document_ids: Optional[List[str]] = None

class QueryResponse(BaseModel):
    success: bool
    query: str
    results: dict
    count: int
    filtered_by_documents: Optional[List[str]] = None
    error: str = None

class DocumentSelectionRequest(BaseModel):
    document_ids: List[str]

class DocumentListResponse(BaseModel):
    success: bool
    documents: List[Dict[str, Any]]
    total_count: int
    error: str = None

class DeleteResponse(BaseModel):
    success: bool
    deleted_documents: int
    deleted_chunks: int
    document_names: List[str]
    message: str
    error: str = None

class UploadResponse(BaseModel):
    success: bool
    file_name: str
    token_count: int
    chunks_created: int
    chunks_deleted: int
    needs_chunking: bool
    action: str
    message: str
    error: str = None

# ===========================
# DATA CLASSES
# ===========================

@dataclass
class ChunkMetadata:
    source: str
    chunk_id: str
    document_hash: str
    document_id: str
    page_number: int = 0
    chunk_index: int = 0
    total_chunks: int = 0
    file_type: str = "unknown"
    upload_timestamp: str = ""
    file_size: int = 0
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary with string values for ChromaDB"""
        return {
            "source": str(self.source),
            "chunk_id": str(self.chunk_id),
            "document_hash": str(self.document_hash),
            "document_id": str(self.document_id),
            "page_number": str(self.page_number),
            "chunk_index": str(self.chunk_index),
            "total_chunks": str(self.total_chunks),
            "file_type": str(self.file_type),
            "upload_timestamp": str(self.upload_timestamp),
            "file_size": str(self.file_size)
        }

# ===========================
# MAIN KB SERVICE CLASS
# ===========================

class CompleteKBService:
    def __init__(
        self, 
        embedding_model_name: str = "jinaai/jina-embeddings-v3",
        max_tokens: int = 8000,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        vector_db_path: str = "./chroma_db",
    ):
        print(f"🔧 Initializing Complete KB Service...")

        # Initialize components
        self.doc_converter = DocumentConverter()
        print(f"📥 Loading embedding model: {embedding_model_name}")

        try:
            self.embedding_model = SentenceTransformer(
                embedding_model_name, trust_remote_code=True
            )
            self.embedding_model.max_seq_length = 8192
        except Exception as e:
            print(f"❌ Failed to load model: {str(e)}")
            print("🔄 Using fallback model...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Configuration
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize ChromaDB
        os.makedirs(vector_db_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=vector_db_path)
        self.collection = self._get_or_create_collection()
        print(f"✅ Complete KB Service initialized successfully!")

    def _get_or_create_collection(self):
        """Get or create ChromaDB collection"""
        try:
            return self.chroma_client.get_collection("knowledge_base")
        except:
            return self.chroma_client.create_collection(
                name="knowledge_base", metadata={"hnsw:space": "cosine"}
            )

    def calculate_document_hash(self, content: str, filename: str) -> str:
        """Calculate hash for document identification"""
        combined = f"{filename}:{content[:1000]}"
        return hashlib.md5(combined.encode()).hexdigest()

    def check_document_exists(self, document_hash: str, filename: str) -> bool:
        """Check if document already exists in vector DB"""
        try:
            results = self.collection.get(
                where={"$or": [{"document_hash": document_hash}, {"source": filename}]}
            )
            return len(results["ids"]) > 0
        except Exception as e:
            print(f"Error checking document existence: {str(e)}")
            return False

    def delete_document(
        self, document_hash: str = None, filename: str = None
    ) -> Dict[str, Any]:
        """Delete all chunks for a specific document"""
        try:
            where_clause = {}
            if document_hash:
                where_clause["document_hash"] = document_hash
            elif filename:
                where_clause["source"] = filename
            else:
                return {"success": False, "error": "No identifier provided"}

            existing_chunks = self.collection.get(where=where_clause)
            chunks_to_delete = len(existing_chunks["ids"])

            if chunks_to_delete > 0:
                self.collection.delete(where=where_clause)
                print(f"🗑️ Deleted {chunks_to_delete} existing chunks")
                return {
                    "success": True,
                    "deleted_chunks": chunks_to_delete,
                    "message": f"Deleted {chunks_to_delete} existing chunks",
                }
            else:
                return {
                    "success": True,
                    "deleted_chunks": 0,
                    "message": "No existing chunks found",
                }

        except Exception as e:
            return {"success": False, "error": f"Failed to delete document: {str(e)}"}

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))

    def parse_document(self, file_path: str) -> Dict[str, Any]:
        """Parse document using Docling"""
        try:
            result = self.doc_converter.convert(file_path)

            content = result.document.export_to_markdown()
            file_type = Path(file_path).suffix.lower()
            filename = Path(file_path).name
            file_size = os.path.getsize(file_path)

            # Calculate document hash for tracking
            doc_hash = self.calculate_document_hash(content, filename)
            doc_id = str(uuid.uuid4())

            metadata = {
                "source": filename,
                "file_path": file_path,
                "file_type": file_type,
                "document_hash": doc_hash,
                "document_id": doc_id,
                "title": getattr(result.document, "title", filename),
                "page_count": (
                    len(result.document.pages)
                    if hasattr(result.document, "pages")
                    else 1
                ),
                "file_size": file_size,
            }

            return {
                "content": content,
                "metadata": metadata,
                "document": result.document,
            }
        except Exception as e:
            raise Exception(f"Failed to parse document {file_path}: {str(e)}")

    def should_chunk(self, content: str) -> bool:
        """Determine if content needs chunking"""
        token_count = self.count_tokens(content)
        return token_count > self.max_tokens

    def create_chunks(
        self, content: str, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create chunks from content"""
        timestamp = datetime.datetime.now().isoformat()

        if not self.should_chunk(content):
            chunk_id = f"{uuid.uuid4().hex[:10]}_0"
            return [
                {
                    "content": content,
                    "metadata": ChunkMetadata(
                        source=metadata["source"],
                        chunk_id=chunk_id,
                        document_hash=metadata["document_hash"],
                        document_id=metadata["document_id"],
                        page_number=metadata.get("page_count", 1),
                        chunk_index=0,
                        total_chunks=1,
                        file_type=metadata.get("file_type", "unknown"),
                        upload_timestamp=timestamp,
                        file_size=metadata.get("file_size", 0),
                    ).to_dict(),
                }
            ]

        # Split into sentences
        sentences = self._split_into_sentences(content)
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunk_content = " ".join(current_chunk)
                chunk_id = f"{uuid.uuid4().hex[:10]}_{chunk_index}"

                chunks.append(
                    {
                        "content": chunk_content,
                        "metadata": ChunkMetadata(
                            source=metadata["source"],
                            chunk_id=chunk_id,
                            document_hash=metadata["document_hash"],
                            document_id=metadata["document_id"],
                            page_number=metadata.get("page_count", 1),
                            chunk_index=chunk_index,
                            total_chunks=0,
                            file_type=metadata.get("file_type", "unknown"),
                            upload_timestamp=timestamp,
                            file_size=metadata.get("file_size", 0),
                        ).to_dict(),
                    }
                )

                # Handle overlap
                overlap_content = self._get_overlap_content(
                    current_chunk, self.chunk_overlap
                )
                current_chunk = overlap_content + [sentence]
                current_tokens = self.count_tokens(" ".join(current_chunk))
                chunk_index += 1
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Add final chunk
        if current_chunk:
            chunk_content = " ".join(current_chunk)
            chunk_id = f"{uuid.uuid4().hex[:10]}_{chunk_index}"

            chunks.append(
                {
                    "content": chunk_content,
                    "metadata": ChunkMetadata(
                        source=metadata["source"],
                        chunk_id=chunk_id,
                        document_hash=metadata["document_hash"],
                        document_id=metadata["document_id"],
                        page_number=metadata.get("page_count", 1),
                        chunk_index=chunk_index,
                        total_chunks=0,
                        file_type=metadata.get("file_type", "unknown"),
                        upload_timestamp=timestamp,
                        file_size=metadata.get("file_size", 0),
                    ).to_dict(),
                }
            )

        # Update total_chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = str(total_chunks)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_content(self, chunks: List[str], overlap_tokens: int) -> List[str]:
        """Get overlap content"""
        if not chunks:
            return []

        overlap_content = []
        token_count = 0

        for sentence in reversed(chunks):
            sentence_tokens = self.count_tokens(sentence)
            if token_count + sentence_tokens <= overlap_tokens:
                overlap_content.insert(0, sentence)
                token_count += sentence_tokens
            else:
                break

        return overlap_content

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings using Jina model"""
        contents = [chunk["content"] for chunk in chunks]
        embeddings = self.embedding_model.encode(contents, convert_to_numpy=True)

        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()

        return chunks

    def store_in_vector_db(self, chunks_with_embeddings: List[Dict[str, Any]]) -> bool:
        """Store in ChromaDB"""
        try:
            ids = [chunk["metadata"]["chunk_id"] for chunk in chunks_with_embeddings]
            documents = [chunk["content"] for chunk in chunks_with_embeddings]
            embeddings = [chunk["embedding"] for chunk in chunks_with_embeddings]

            # Ensure all metadata values are strings and non-None
            metadatas = []
            for chunk in chunks_with_embeddings:
                clean_metadata = {}
                for key, value in chunk["metadata"].items():
                    if value is not None:
                        clean_metadata[key] = str(value)
                    else:
                        clean_metadata[key] = (
                            "0"
                            if key in ["page_number", "chunk_index", "total_chunks"]
                            else "unknown"
                        )
                metadatas.append(clean_metadata)

            self.collection.add(
                ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
            )

            return True
        except Exception as e:
            print(f"Error storing in vector DB: {str(e)}")
            return False

    def process_document(
        self, file_path: str, replace_existing: bool = True
    ) -> Dict[str, Any]:
        """Complete pipeline with document replacement support"""
        try:
            # Step 1: Parse document
            print(f"📄 Parsing document: {Path(file_path).name}")
            parsed_doc = self.parse_document(file_path)

            content = parsed_doc["content"]
            metadata = parsed_doc["metadata"]
            filename = metadata["source"]
            doc_hash = metadata["document_hash"]

            # Step 2: Check if document exists
            document_exists = self.check_document_exists(doc_hash, filename)
            deleted_chunks = 0

            if document_exists and replace_existing:
                print(f"🔄 Document exists, replacing...")
                delete_result = self.delete_document(
                    document_hash=doc_hash, filename=filename
                )
                if delete_result["success"]:
                    deleted_chunks = delete_result["deleted_chunks"]
                    print(f"🗑️ Removed {deleted_chunks} existing chunks")
            elif document_exists and not replace_existing:
                return {
                    "success": False,
                    "error": f"Document '{filename}' already exists. Set replace_existing=True to overwrite.",
                    "file_name": filename,
                    "action": "skipped",
                }

            # Step 3: Process new document
            token_count = self.count_tokens(content)
            needs_chunking = self.should_chunk(content)

            chunks = self.create_chunks(content, metadata)
            print(f"📝 Created {len(chunks)} new chunks")

            # Step 4: Generate embeddings
            print("🧮 Generating embeddings...")
            chunks_with_embeddings = self.generate_embeddings(chunks)

            # Step 5: Store in vector DB
            print("💾 Storing in vector database...")
            success = self.store_in_vector_db(chunks_with_embeddings)

            action = "replaced" if document_exists else "added"

            return {
                "success": success,
                "file_name": filename,
                "token_count": token_count,
                "chunks_created": len(chunks),
                "chunks_deleted": deleted_chunks,
                "needs_chunking": needs_chunking,
                "action": action,
                "message": f"Successfully {action} '{filename}' with {len(chunks)} chunks"
                + (
                    f" (removed {deleted_chunks} old chunks)"
                    if deleted_chunks > 0
                    else ""
                ),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_name": Path(file_path).name if file_path else "Unknown",
            }

    def query_knowledge_base(self, query: str, n_results: int = 5, document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Query the knowledge base with optional document filtering"""
        try:
            query_embedding = self.embedding_model.encode([query])
            
            # Build where clause for document filtering
            where_clause = None
            if document_ids:
                where_clause = {"document_id": {"$in": document_ids}}

            results = self.collection.query(
                query_embeddings=query_embedding.tolist(), 
                n_results=n_results,
                where=where_clause
            )

            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results["documents"][0]) if results["documents"] else 0,
                "filtered_by_documents": document_ids if document_ids else None,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "query": query}

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()

            # Get unique documents
            all_docs = self.collection.get()
            unique_sources = set()
            if all_docs["metadatas"]:
                unique_sources = {
                    meta.get("source", "Unknown") for meta in all_docs["metadatas"]
                }

            return {
                "total_chunks": count,
                "unique_documents": len(unique_sources),
                "documents": list(unique_sources),
                "collection_name": "knowledge_base",
            }
        except Exception as e:
            return {"error": str(e)}

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the database"""
        try:
            all_docs = self.collection.get()
            if not all_docs["metadatas"]:
                return []

            # Group by document
            doc_info = {}
            for metadata in all_docs["metadatas"]:
                source = metadata.get("source", "Unknown")
                if source not in doc_info:
                    doc_info[source] = {
                        "filename": source,
                        "chunks": 0,
                        "file_type": metadata.get("file_type", "unknown"),
                        "upload_timestamp": metadata.get("upload_timestamp", "unknown"),
                        "document_hash": metadata.get("document_hash", "unknown"),
                        "document_id": metadata.get("document_id", "unknown"),
                        "file_size": int(metadata.get("file_size", 0)),
                    }
                doc_info[source]["chunks"] += 1

            return list(doc_info.values())
        except Exception as e:
            return []

    def delete_documents_by_ids(self, document_ids: List[str]) -> Dict[str, Any]:
        """Delete documents by their IDs"""
        try:
            total_deleted_chunks = 0
            deleted_documents = 0
            deleted_document_names = []

            for doc_id in document_ids:
                # Get chunks for this document
                existing_chunks = self.collection.get(where={"document_id": doc_id})
                chunks_to_delete = len(existing_chunks["ids"])

                if chunks_to_delete > 0:
                    # Get document name before deletion
                    if existing_chunks["metadatas"]:
                        doc_name = existing_chunks["metadatas"][0].get("source", "Unknown")
                        deleted_document_names.append(doc_name)

                    # Delete chunks
                    self.collection.delete(where={"document_id": doc_id})
                    total_deleted_chunks += chunks_to_delete
                    deleted_documents += 1

            return {
                "success": True,
                "deleted_documents": deleted_documents,
                "deleted_chunks": total_deleted_chunks,
                "document_names": deleted_document_names,
                "message": f"Deleted {deleted_documents} documents with {total_deleted_chunks} chunks",
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to delete documents: {str(e)}"}

# ===========================
# FASTAPI APPLICATION
# ===========================

# Initialize the KB service
print("🚀 Starting Complete Knowledge Base API...")
kb_service = CompleteKBService()

# Create FastAPI app
app = FastAPI(
    title="Knowledge Base API",
    description="Complete Knowledge Base REST API with document management and querying",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# API ROUTES
# ===========================

@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document via API"""
    try:
        # Validate file type
        allowed_extensions = {".pdf", ".docx", ".doc", ".txt", ".md"}
        file_extension = os.path.splitext(file.filename)[1].lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type: {file_extension}"
            )

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension
        ) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Process document
            result = kb_service.process_document(temp_file_path)
            return UploadResponse(**result)
        finally:
            # Clean up temp file
            os.unlink(temp_file_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the knowledge base via API"""
    try:
        result = kb_service.query_knowledge_base(
            request.query, 
            request.n_results, 
            request.document_ids
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Get collection statistics via API"""
    return kb_service.get_collection_stats()

@app.get("/api/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all documents in the database"""
    try:
        documents = kb_service.list_documents()
        return DocumentListResponse(
            success=True,
            documents=documents,
            total_count=len(documents)
        )
    except Exception as e:
        return DocumentListResponse(
            success=False,
            documents=[],
            total_count=0,
            error=str(e)
        )

@app.delete("/api/documents", response_model=DeleteResponse)
async def delete_documents(request: DocumentSelectionRequest):
    """Delete documents by their IDs"""
    try:
        result = kb_service.delete_documents_by_ids(request.document_ids)
        return DeleteResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Knowledge Base API"}

if __name__ == "__main__":
    import uvicorn

    print("🌟 Starting Complete Knowledge Base API on http://localhost:8000")
    print("🔗 API Docs: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")