# complete_advanced_kb_system.py
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

import gradio as gr
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
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
        vector_db_path: str = "./chroma_db"
    ):
        print(f"🔧 Initializing Complete KB Service...")
        
        # Initialize components
        self.doc_converter = DocumentConverter()
        print(f"📥 Loading embedding model: {embedding_model_name}")
        
        try:
            self.embedding_model = SentenceTransformer(
                embedding_model_name, 
                trust_remote_code=True
            )
            self.embedding_model.max_seq_length = 8192
            print(f"✅ Successfully loaded {embedding_model_name}")
        except Exception as e:
            print(f"❌ Failed to load {embedding_model_name}: {str(e)}")
            print("🔄 Using fallback model...")
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Using fallback model: all-MiniLM-L6-v2")
        
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
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )
    
    def generate_document_id(self) -> str:
        """Generate unique document ID"""
        return f"doc_{uuid.uuid4().hex[:12]}"
    
    def calculate_document_hash(self, content: str, filename: str) -> str:
        """Calculate hash for document identification"""
        combined = f"{filename}:{content[:1000]}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get_file_size(self, file_path: str) -> int:
        """Get file size in bytes"""
        try:
            return os.path.getsize(file_path)
        except:
            return 0
    
    # ===========================
    # VECTOR DB MANAGEMENT
    # ===========================
    
    def clear_all_data(self) -> Dict[str, Any]:
        """Clear all data from vector database"""
        try:
            current_count = self.collection.count()
            
            if current_count == 0:
                return {
                    "success": True,
                    "deleted_chunks": 0,
                    "message": "Database is already empty"
                }
            
            # Delete all documents
            all_docs = self.collection.get()
            if all_docs["ids"]:
                self.collection.delete(ids=all_docs["ids"])
            
            return {
                "success": True,
                "deleted_chunks": current_count,
                "message": f"Successfully cleared {current_count} chunks from vector database"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to clear database: {str(e)}"
            }
    
    def delete_documents_by_ids(self, document_ids: List[str]) -> Dict[str, Any]:
        """Delete specific documents by their IDs"""
        try:
            deleted_count = 0
            deleted_docs = []
            failed_deletions = []
            
            for doc_id in document_ids:
                try:
                    # Get chunks for this document
                    doc_chunks = self.collection.get(where={"document_id": doc_id})
                    
                    if doc_chunks["ids"]:
                        # Delete chunks
                        self.collection.delete(where={"document_id": doc_id})
                        deleted_count += len(doc_chunks["ids"])
                        
                        # Get document name
                        if doc_chunks["metadatas"]:
                            doc_name = doc_chunks["metadatas"][0].get("source", doc_id)
                            deleted_docs.append(doc_name)
                    else:
                        failed_deletions.append(f"Document {doc_id} not found")
                except Exception as e:
                    failed_deletions.append(f"Failed to delete {doc_id}: {str(e)}")
            
            return {
                "success": True,
                "deleted_documents": len(deleted_docs),
                "deleted_chunks": deleted_count,
                "document_names": deleted_docs,
                "failed_deletions": failed_deletions,
                "message": f"Successfully deleted {len(deleted_docs)} documents ({deleted_count} chunks)"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to delete documents: {str(e)}"
            }
    
    def list_all_documents(self, include_chunks: bool = False) -> Dict[str, Any]:
        """List all documents with their metadata and IDs"""
        try:
            all_docs = self.collection.get()
            if not all_docs["metadatas"]:
                return {
                    "success": True,
                    "documents": [],
                    "total_count": 0
                }
            
            # Group by document ID
            doc_info = {}
            for i, metadata in enumerate(all_docs["metadatas"]):
                doc_id = metadata.get("document_id", "unknown")
                source = metadata.get("source", "Unknown")
                
                if doc_id not in doc_info:
                    doc_info[doc_id] = {
                        "document_id": doc_id,
                        "filename": source,
                        "chunks": 0,
                        "file_type": metadata.get("file_type", "unknown"),
                        "upload_timestamp": metadata.get("upload_timestamp", "unknown"),
                        "document_hash": metadata.get("document_hash", "unknown"),
                        "file_size": int(metadata.get("file_size", 0)),
                        "total_tokens": 0
                    }
                
                doc_info[doc_id]["chunks"] += 1
                
                # Add chunk info if requested
                if include_chunks:
                    if "chunk_details" not in doc_info[doc_id]:
                        doc_info[doc_id]["chunk_details"] = []
                    
                    doc_info[doc_id]["chunk_details"].append({
                        "chunk_id": metadata.get("chunk_id", "unknown"),
                        "chunk_index": int(metadata.get("chunk_index", 0)),
                        "content_preview": all_docs["documents"][i][:100] + "..." if len(all_docs["documents"][i]) > 100 else all_docs["documents"][i]
                    })
            
            # Sort by upload timestamp (newest first)
            documents = sorted(doc_info.values(), key=lambda x: x["upload_timestamp"], reverse=True)
            
            return {
                "success": True,
                "documents": documents,
                "total_count": len(documents)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to list documents: {str(e)}",
                "documents": [],
                "total_count": 0
            }
    
    def get_document_by_id(self, document_id: str) -> Dict[str, Any]:
        """Get specific document by ID with all its chunks"""
        try:
            doc_chunks = self.collection.get(where={"document_id": document_id})
            
            if not doc_chunks["ids"]:
                return {
                    "success": False,
                    "error": f"Document with ID '{document_id}' not found"
                }
            
            # Organize chunks
            chunks = []
            for i in range(len(doc_chunks["ids"])):
                chunks.append({
                    "chunk_id": doc_chunks["ids"][i],
                    "content": doc_chunks["documents"][i],
                    "metadata": doc_chunks["metadatas"][i],
                    "chunk_index": int(doc_chunks["metadatas"][i].get("chunk_index", 0))
                })
            
            # Sort by chunk index
            chunks.sort(key=lambda x: x["chunk_index"])
            
            # Get document info
            first_chunk_meta = doc_chunks["metadatas"][0]
            
            return {
                "success": True,
                "document_id": document_id,
                "filename": first_chunk_meta.get("source", "Unknown"),
                "file_type": first_chunk_meta.get("file_type", "unknown"),
                "upload_timestamp": first_chunk_meta.get("upload_timestamp", "unknown"),
                "file_size": int(first_chunk_meta.get("file_size", 0)),
                "total_chunks": len(chunks),
                "chunks": chunks,
                "full_content": " ".join([chunk["content"] for chunk in chunks])
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get document: {str(e)}"
            }
    
    # ===========================
    # DOCUMENT PROCESSING
    # ===========================
    
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
            file_size = self.get_file_size(file_path)
            
            # Generate unique document ID and hash
            doc_id = self.generate_document_id()
            doc_hash = self.calculate_document_hash(content, filename)
            
            metadata = {
                "source": filename,
                "file_path": file_path,
                "file_type": file_type,
                "document_id": doc_id,
                "document_hash": doc_hash,
                "file_size": file_size,
                "title": getattr(result.document, 'title', filename),
                "page_count": len(result.document.pages) if hasattr(result.document, 'pages') else 1
            }
            
            return {
                "content": content,
                "metadata": metadata,
                "document": result.document
            }
        except Exception as e:
            raise Exception(f"Failed to parse document {file_path}: {str(e)}")
    
    def should_chunk(self, content: str) -> bool:
        """Determine if content needs chunking"""
        token_count = self.count_tokens(content)
        return token_count > self.max_tokens
    
    def create_chunks(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks from content"""
        timestamp = datetime.datetime.now().isoformat()
        
        if not self.should_chunk(content):
            chunk_id = f"{uuid.uuid4().hex[:10]}_0"
            return [{
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
                    file_size=metadata.get("file_size", 0)
                ).to_dict()
            }]
        
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
                
                chunks.append({
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
                        file_size=metadata.get("file_size", 0)
                    ).to_dict()
                })
                
                # Handle overlap
                overlap_content = self._get_overlap_content(current_chunk, self.chunk_overlap)
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
            
            chunks.append({
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
                    file_size=metadata.get("file_size", 0)
                ).to_dict()
            })
        
        # Update total_chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = str(total_chunks)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
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
        """Generate embeddings using embedding model"""
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
                        clean_metadata[key] = "0" if key in ["page_number", "chunk_index", "total_chunks", "file_size"] else "unknown"
                metadatas.append(clean_metadata)
            
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            return True
        except Exception as e:
            print(f"Error storing in vector DB: {str(e)}")
            return False
    
    def check_document_exists(self, document_hash: str, filename: str) -> bool:
        """Check if document already exists in vector DB"""
        try:
            results = self.collection.get(
                where={"$or": [
                    {"document_hash": document_hash},
                    {"source": filename}
                ]}
            )
            return len(results['ids']) > 0
        except Exception as e:
            print(f"Error checking document existence: {str(e)}")
            return False
    
    def delete_document(self, document_hash: str = None, filename: str = None) -> Dict[str, Any]:
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
            chunks_to_delete = len(existing_chunks['ids'])
            
            if chunks_to_delete > 0:
                self.collection.delete(where=where_clause)
                print(f"🗑️ Deleted {chunks_to_delete} existing chunks")
                return {
                    "success": True, 
                    "deleted_chunks": chunks_to_delete,
                    "message": f"Deleted {chunks_to_delete} existing chunks"
                }
            else:
                return {
                    "success": True, 
                    "deleted_chunks": 0,
                    "message": "No existing chunks found"
                }
                
        except Exception as e:
            return {
                "success": False, 
                "error": f"Failed to delete document: {str(e)}"
            }
    
    def process_document(self, file_path: str, replace_existing: bool = True) -> Dict[str, Any]:
        """Complete pipeline with document replacement support"""
        try:
            print(f"📄 Parsing document: {Path(file_path).name}")
            parsed_doc = self.parse_document(file_path)
            
            content = parsed_doc["content"]
            metadata = parsed_doc["metadata"]
            filename = metadata["source"]
            doc_hash = metadata["document_hash"]
            doc_id = metadata["document_id"]
            
            # Check if document exists
            document_exists = self.check_document_exists(doc_hash, filename)
            deleted_chunks = 0
            
            if document_exists and replace_existing:
                print(f"🔄 Document exists, replacing...")
                delete_result = self.delete_document(document_hash=doc_hash, filename=filename)
                if delete_result["success"]:
                    deleted_chunks = delete_result["deleted_chunks"]
                    print(f"🗑️ Removed {deleted_chunks} existing chunks")
            elif document_exists and not replace_existing:
                return {
                    "success": False,
                    "error": f"Document '{filename}' already exists. Set replace_existing=True to overwrite.",
                    "file_name": filename,
                    "action": "skipped"
                }
            
            # Process new document
            token_count = self.count_tokens(content)
            needs_chunking = self.should_chunk(content)
            
            chunks = self.create_chunks(content, metadata)
            print(f"📝 Created {len(chunks)} new chunks")
            
            # Generate embeddings
            print("🧮 Generating embeddings...")
            chunks_with_embeddings = self.generate_embeddings(chunks)
            
            # Store in vector DB
            print("💾 Storing in vector database...")
            success = self.store_in_vector_db(chunks_with_embeddings)
            
            action = "replaced" if document_exists else "added"
            
            return {
                "success": success,
                "file_name": filename,
                "document_id": doc_id,
                "token_count": token_count,
                "chunks_created": len(chunks),
                "chunks_deleted": deleted_chunks,
                "needs_chunking": needs_chunking,
                "action": action,
                "file_size": metadata["file_size"],
                "message": f"Successfully {action} '{filename}' with {len(chunks)} chunks" + 
                          (f" (removed {deleted_chunks} old chunks)" if deleted_chunks > 0 else "")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_name": Path(file_path).name if file_path else "Unknown"
            }
    
    # ===========================
    # QUERY OPERATIONS
    # ===========================
    
    def query_knowledge_base(self, query: str, n_results: int = 5, document_ids: List[str] = None) -> Dict[str, Any]:
        """Query the knowledge base with optional document filtering"""
        try:
            query_embedding = self.embedding_model.encode([query])
            
            # Build where clause for document filtering
            where_clause = None
            if document_ids:
                if len(document_ids) == 1:
                    where_clause = {"document_id": document_ids[0]}
                else:
                    where_clause = {"$or": [{"document_id": doc_id} for doc_id in document_ids]}
            
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
                "filtered_by_documents": document_ids if document_ids else None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            
            # Get unique documents
            all_docs = self.collection.get()
            unique_sources = set()
            unique_doc_ids = set()
            total_file_size = 0
            
            if all_docs["metadatas"]:
                unique_sources = {meta.get("source", "Unknown") for meta in all_docs["metadatas"]}
                unique_doc_ids = {meta.get("document_id", "Unknown") for meta in all_docs["metadatas"]}
                
                # Calculate total file size (sum unique documents only)
                doc_sizes = {}
                for meta in all_docs["metadatas"]:
                    doc_id = meta.get("document_id", "Unknown")
                    if doc_id not in doc_sizes:
                        doc_sizes[doc_id] = int(meta.get("file_size", 0))
                total_file_size = sum(doc_sizes.values())
            
            return {
                "total_chunks": count,
                "unique_documents": len(unique_sources),
                "unique_document_ids": len(unique_doc_ids),
                "total_file_size": total_file_size,
                "documents": list(unique_sources),
                "collection_name": "knowledge_base"
            }
        except Exception as e:
            return {"error": str(e)}

# ===========================
# INITIALIZE KB SERVICE
# ===========================

print("🚀 Starting Complete Knowledge Base System...")
kb_service = CompleteKBService()

# ===========================
# FASTAPI APPLICATION
# ===========================

app = FastAPI(
    title="Complete Knowledge Base API",
    description="Advanced Knowledge Base with Document Management, Vector Operations, and Selection Features",
    version="3.0.0"
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
# API ENDPOINTS
# ===========================

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document via API"""
    try:
        allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md'}
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}. Allowed: {', '.join(allowed_extensions)}"
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            result = kb_service.process_document(temp_file_path)
            return result
        finally:
            os.unlink(temp_file_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the knowledge base via API with optional document filtering"""
    try:
        result = kb_service.query_knowledge_base(
            request.query, 
            request.n_results,
            request.document_ids
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents", response_model=DocumentListResponse)
async def list_documents(include_chunks: bool = Query(False, description="Include chunk details")):
    """Get list of all documents with their IDs and metadata"""
    try:
        result = kb_service.list_all_documents(include_chunks=include_chunks)
        return DocumentListResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/{document_id}")
async def get_document_by_id(document_id: str):
    """Get specific document by ID with all chunks and content"""
    try:
        result = kb_service.get_document_by_id(document_id)
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents")
async def delete_documents(request: DocumentSelectionRequest):
    """Delete specific documents by their IDs"""
    try:
        if not request.document_ids:
            raise HTTPException(status_code=400, detail="No document IDs provided")
        
        result = kb_service.delete_documents_by_ids(request.document_ids)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/clear-all")
async def clear_all_data():
    """Clear all data from vector database"""
    try:
        result = kb_service.clear_all_data()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Get comprehensive collection statistics"""
    return kb_service.get_collection_stats()

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Complete Knowledge Base API",
        "version": "3.0.0",
        "features": [
            "document_upload",
            "vector_search", 
            "document_management",
            "bulk_operations",
            "selective_querying"
        ]
    }

# ===========================
# GRADIO INTERFACE FUNCTIONS
# ===========================

def upload_and_process(file):
    """Upload and process document via Gradio"""
    if file is None:
        return "Please select a file to upload.", "", []
    
    try:
        result = kb_service.process_document(file.name, replace_existing=True)
        
        if result["success"]:
            stats = kb_service.get_collection_stats()
            
            action_emoji = "🔄" if result["action"] == "replaced" else "✅"
            action_text = "Replaced" if result["action"] == "replaced" else "Added"
            
            status_msg = f"""
            {action_emoji} **{action_text}!** 
            - File: {result['file_name']}
            - Document ID: `{result['document_id']}`
            - Action: {result['action'].title()}
            - File Size: {result['file_size']:,} bytes
            - Tokens: {result['token_count']:,}
            - New chunks: {result['chunks_created']}
            {f"- Deleted old chunks: {result['chunks_deleted']}" if result.get('chunks_deleted', 0) > 0 else ""}
            - Total chunks in DB: {stats.get('total_chunks', 0)}
            - Unique documents: {stats.get('unique_documents', 0)}
            """
            
            # Update document list
            doc_list = refresh_document_list()
            return status_msg, "", doc_list
        else:
            return f"❌ **Error:** {result['error']}", "", []
            
    except Exception as e:
        return f"❌ **Error:** {str(e)}", "", []

def query_documents(query, n_results, selected_docs):
    """Query the knowledge base via Gradio"""
    if not query.strip():
        return "Please enter a query.", []
    
    try:
        # Get selected document IDs
        document_ids = None
        if selected_docs:
            # Extract document IDs from selected rows
            document_ids = [doc[1] for doc in selected_docs]  # Column 1 is document_id
        
        result = kb_service.query_knowledge_base(query, n_results, document_ids)
        
        if result["success"] and result["count"] > 0:
            formatted_results = []
            documents = result["results"]["documents"][0]
            metadatas = result["results"]["metadatas"][0]
            distances = result["results"]["distances"][0]
            
            for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                try:
                    chunk_index = int(metadata.get('chunk_index', '0'))
                    total_chunks = int(metadata.get('total_chunks', '1'))
                    source = str(metadata.get("source", "Unknown"))
                    doc_id = str(metadata.get("document_id", "Unknown"))
                    distance_float = float(distance)
                    
                    formatted_results.append([
                        i + 1,
                        source,
                        doc_id[:12] + "...",  # Truncate ID for display
                        f"{chunk_index + 1}/{total_chunks}",
                        f"{1 - distance_float:.3f}",
                        doc[:300] + "..." if len(doc) > 300 else doc
                    ])
                except (ValueError, TypeError):
                    formatted_results.append([
                        i + 1,
                        str(metadata.get("source", "Unknown")),
                        str(metadata.get("document_id", "Unknown"))[:12] + "...",
                        "1/1",
                        f"{1 - float(distance):.3f}",
                        doc[:300] + "..." if len(doc) > 300 else doc
                    ])
            
            filter_msg = f" (filtered by {len(document_ids)} documents)" if document_ids else ""
            return f"Found {result['count']} relevant results{filter_msg}:", formatted_results
        else:
            return "No relevant documents found.", []
            
    except Exception as e:
        return f"❌ **Error:** {str(e)}", []

def refresh_document_list():
    """Refresh the document list"""
    try:
        result = kb_service.list_all_documents()
        if result["success"]:
            doc_data = []
            for doc in result["documents"]:
                # Format file size
                size_mb = doc["file_size"] / (1024 * 1024)
                size_str = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{doc['file_size']:,} bytes"
                
                # Format timestamp
                timestamp = doc["upload_timestamp"][:19] if len(doc["upload_timestamp"]) > 19 else doc["upload_timestamp"]
                
                doc_data.append([
                    doc["filename"],
                    doc["document_id"],
                    doc["chunks"],
                    doc["file_type"],
                    size_str,
                    timestamp
                ])
            return doc_data
        else:
            return []
    except Exception as e:
        print(f"Error refreshing document list: {str(e)}")
        return []

def delete_selected_documents(selected_docs):
    """Delete selected documents"""
    if not selected_docs:
        return "Please select documents to delete.", refresh_document_list()
    
    try:
        # Extract document IDs from selected rows
        document_ids = [doc[1] for doc in selected_docs]  # Column 1 is document_id
        
        result = kb_service.delete_documents_by_ids(document_ids)
        
        if result["success"]:
            failed_msg = ""
            if result.get("failed_deletions"):
                failed_msg = f"\n\n⚠️ **Some deletions failed:**\n" + "\n".join(result["failed_deletions"])
            
            status_msg = f"""
            🗑️ **Deleted Successfully!**
            - Documents deleted: {result['deleted_documents']}
            - Chunks deleted: {result['deleted_chunks']}
            - Files: {', '.join(result['document_names'])}
            {failed_msg}
            """
        else:
            status_msg = f"❌ **Error:** {result['error']}"
        
        # Refresh document list
        updated_list = refresh_document_list()
        return status_msg, updated_list
        
    except Exception as e:
        return f"❌ **Error:** {str(e)}", refresh_document_list()

def clear_all_data():
    """Clear all data from vector database"""
    try:
        result = kb_service.clear_all_data()
        
        if result["success"]:
            status_msg = f"""
            🧹 **Database Cleared!**
            - Deleted chunks: {result['deleted_chunks']}
            - Database is now empty
            - All documents and vectors have been removed
            """
        else:
            status_msg = f"❌ **Error:** {result['error']}"
        
        # Refresh document list (should be empty)
        updated_list = refresh_document_list()
        return status_msg, updated_list
        
    except Exception as e:
        return f"❌ **Error:** {str(e)}", refresh_document_list()

def get_database_stats():
    """Get enhanced database statistics"""
    try:
        stats = kb_service.get_collection_stats()
        result = kb_service.list_all_documents()
        
        # Format file size
        total_size = stats.get('total_file_size', 0)
        size_mb = total_size / (1024 * 1024)
        size_str = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{total_size:,} bytes"
        
        doc_list = ""
        if result["success"] and result["documents"]:
            doc_list = "\n\n**Recent Documents:**\n" + "\n".join([
                f"- {doc['filename']} ({doc['chunks']} chunks, ID: {doc['document_id'][:8]}...)" 
                for doc in result["documents"][:10]
            ])
            if len(result["documents"]) > 10:
                doc_list += f"\n... and {len(result['documents']) - 10} more"
        
        return f"""📊 **Database Statistics:**
- Total chunks: {stats.get('total_chunks', 0):,}
- Unique documents: {stats.get('unique_documents', 0):,}
- Document IDs: {stats.get('unique_document_ids', 0):,}
- Total file size: {size_str}
- Collection: {stats.get('collection_name', 'knowledge_base')}
{doc_list}
"""
    except Exception as e:
        return f"❌ **Error:** {str(e)}"

# ===========================
# GRADIO INTERFACE
# ===========================

with gr.Blocks(
    title="Complete Knowledge Base System",
    theme=gr.themes.Soft(),
    css="""
        .gradio-container {
            max-width: 1600px !important;
        }
        .main-header {
            text-align: center;
            padding: 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .danger-button {
            background: #ff4757 !important;
            color: white !important;
        }
        .success-button {
            background: #2ed573 !important;
            color: white !important;
        }
    """
) as demo:
    
    gr.HTML("""
        <div class="main-header">
            <h1>🧠 Complete Knowledge Base System</h1>
            <p>Advanced document management with vector operations, selective querying, and comprehensive API</p>
            <p><strong>Features:</strong> Upload • Query • Manage • Delete • Vector Operations • Document Selection</p>
        </div>
    """)
    
    with gr.Tabs():
        with gr.Tab("📤 Upload Documents"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(
                        label="📄 Select Document",
                        file_types=[".pdf", ".docx", ".doc", ".txt", ".md"],
                        file_count="single"
                    )
                    upload_btn = gr.Button("🚀 Upload & Process", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    upload_status = gr.Markdown("Ready to upload documents...")
            
            with gr.Row():
                upload_document_list = gr.Dataframe(
                    headers=["Filename", "Document ID", "Chunks", "Type", "Size", "Uploaded"],
                    datatype=["str", "str", "number", "str", "str", "str"],
                    label="📋 Uploaded Documents",
                    interactive=False,
                    value=refresh_document_list(),
                    wrap=True
                )
                    
            upload_btn.click(
                upload_and_process,
                inputs=[file_input],
                outputs=[upload_status, gr.Textbox(visible=False), upload_document_list]
            )
        
        with gr.Tab("🔍 Query Knowledge Base"):
            with gr.Row():
                with gr.Column(scale=2):
                    query_input = gr.Textbox(
                        label="💬 Enter your question",
                        placeholder="What would you like to know?",
                        lines=4
                    )
                
                with gr.Column(scale=1):
                    n_results = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="📊 Number of results"
                    )
                    query_btn = gr.Button("🔍 Search", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📋 Select Documents to Search (Optional)")
                    gr.Markdown("💡 **Tip:** Select specific documents to limit your search scope, or leave empty to search all documents.")
                    
                    document_selector = gr.Dataframe(
                        headers=["Filename", "Document ID", "Chunks", "Type", "Size", "Uploaded"],
                        datatype=["str", "str", "number", "str", "str", "str"],
                        label="🎯 Available Documents (Select rows to filter search)",
                        interactive=True,
                        value=refresh_document_list(),
                        wrap=True
                    )
            
            query_status = gr.Markdown("Enter a question to search...")
            
            results_table = gr.Dataframe(
                headers=["#", "Source", "Doc ID", "Chunk", "Similarity", "Content"],
                datatype=["number", "str", "str", "str", "str", "str"],
                label="🎯 Search Results",
                interactive=False,
                wrap=True
            )
            
            query_btn.click(
                query_documents,
                inputs=[query_input, n_results, document_selector],
                outputs=[query_status, results_table]
            )
        
        with gr.Tab("🗂️ Document Management"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📋 Document List")
                    refresh_btn = gr.Button("🔄 Refresh List", variant="secondary")
                    
                    management_document_list = gr.Dataframe(
                        headers=["Filename", "Document ID", "Chunks", "Type", "Size", "Uploaded"],
                        datatype=["str", "str", "number", "str", "str", "str"],
                        label="📚 All Documents (Select rows for bulk operations)",
                        interactive=True,
                        value=refresh_document_list(),
                        wrap=True
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🎛️ Management Actions")
                    
                    gr.Markdown("#### 🗑️ Delete Operations")
                    delete_selected_btn = gr.Button("🗑️ Delete Selected Documents", variant="stop")
                    
                    gr.Markdown("---")
                    
                    gr.Markdown("#### ⚠️ Danger Zone")
                    gr.Markdown("⚠️ **Warning:** This will permanently delete ALL data!")
                    clear_all_btn = gr.Button("🧹 Clear Entire Database", elem_classes=["danger-button"])
                    
                    gr.Markdown("---")
                    
                    management_status = gr.Markdown("Select documents and choose an action...")
            
            refresh_btn.click(
                refresh_document_list,
                outputs=[management_document_list]
            )
            
            delete_selected_btn.click(
                delete_selected_documents,
                inputs=[management_document_list],
                outputs=[management_status, management_document_list]
            )
            
            clear_all_btn.click(
                clear_all_data,
                outputs=[management_status, management_document_list]
            )
        
        with gr.Tab("📊 Database Statistics"):
            with gr.Row():
                with gr.Column():
                    stats_btn = gr.Button("📊 Refresh Statistics", variant="secondary", size="lg")
                    stats_output = gr.Markdown("Click to view comprehensive database statistics...")
                
                with gr.Column():
                    gr.Markdown("""
                    ### 📈 System Information
                    
                    **🔧 Technical Details:**
                    - **Embedding Model**: Jina 8K Context (v3) with fallback
                    - **Vector Database**: ChromaDB (Persistent storage)
                    - **Max Token Limit**: 8,000 per direct embedding
                    - **Chunk Size**: 1,000 tokens with 200 token overlap
                    - **Supported Formats**: PDF, DOCX, DOC, TXT, MD
                    
                    **🎯 Key Features:**
                    - Document replacement and versioning
                    - Unique document ID tracking
                    - Selective querying by document
                    - Bulk operations and management
                    - Real-time statistics and monitoring
                    - Complete REST API integration
                    """)
            
            stats_btn.click(get_database_stats, outputs=[stats_output])
    
    gr.Markdown("""
    ---
    ## 🚀 Complete REST API Reference
    
    **Base URL:** `http://localhost:7860`
    
    ### 📋 Core Endpoints
    
    | Method | Endpoint | Description | Request Body |
    |--------|----------|-------------|--------------|
    | POST | `/api/upload` | Upload document | `multipart/form-data` |
    | POST | `/api/query` | Query with filtering | `{"query": str, "n_results": int, "document_ids": [str]}` |
    | GET | `/api/documents` | List all documents | `?include_chunks=bool` |
    | GET | `/api/documents/{doc_id}` | Get specific document | - |
    | DELETE | `/api/documents` | Delete documents | `{"document_ids": [str]}` |
    | DELETE | `/api/clear-all` | Clear database | - |
    | GET | `/api/stats` | Get statistics | - |
    | GET | `/api/health` | Health check | - |
    
    ### 📖 Interactive Documentation
    **Complete API Docs:** [http://localhost:7860/docs](http://localhost:7860/docs)
    
    ### 💡 Usage Examples
    
    ```
    # Upload document
    curl -X POST "http://localhost:7860/api/upload" \\
         -F "file=@document.pdf"
    
    # List all documents
    curl -X GET "http://localhost:7860/api/documents"
    
    # Query specific documents
    curl -X POST "http://localhost:7860/api/query" \\
         -H "Content-Type: application/json" \\
         -d '{
           "query": "What is artificial intelligence?",
           "document_ids": ["doc_abc123456789", "doc_def987654321"],
           "n_results": 10
         }'
    
    # Get document by ID
    curl -X GET "http://localhost:7860/api/documents/doc_abc123456789"
    
    # Delete specific documents
    curl -X DELETE "http://localhost:7860/api/documents" \\
         -H "Content-Type: application/json" \\
         -d '{"document_ids": ["doc_abc123456789", "doc_def987654321"]}'
    
    # Clear entire database
    curl -X DELETE "http://localhost:7860/api/clear-all"
    
    # Get database statistics
    curl -X GET "http://localhost:7860/api/stats"
    ```
    
    ### 🎯 Advanced Features
    - **Document Selection**: Filter queries by specific document IDs
    - **Bulk Operations**: Upload, delete, and manage multiple documents
    - **Vector Management**: Complete control over vector database
    - **Real-time Stats**: Monitor usage and performance
    - **Error Handling**: Comprehensive error responses
    - **CORS Enabled**: Ready for web application integration
    
    ---
    **🔧 Built with:** FastAPI • Gradio • Docling • Jina Embeddings • ChromaDB • Sentence Transformers
    """)

# ===========================
# APPLICATION STARTUP
# ===========================

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Check if running on HF Spaces
    is_hf_space = os.getenv("SPACE_ID") is not None
    
    if is_hf_space:
        print("🚀 Running on Hugging Face Spaces")
        # For HF Spaces, mount Gradio on root path with specific config
        app = gr.mount_gradio_app(
            app, 
            demo, 
            path="/",
            app_kwargs={"docs_url": "/api/docs", "redoc_url": "/api/redoc"}
        )
    else:
        print("🚀 Running locally")
        # For local development
        app = gr.mount_gradio_app(app, demo, path="/")
    
    print("🌟 Starting Advanced Knowledge Base System")
    print("📖 Gradio UI: Available on root path")
    print("🔗 API Docs: /api/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        log_level="info"
    )
