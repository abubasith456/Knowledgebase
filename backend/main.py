import os
import hashlib
import tempfile
import threading
import time
from typing import List, Dict, Any, Optional, Literal
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
    indexing_mode: str
    embedding_model: str
    error: str = None

class IndexingConfig(BaseModel):
    mode: Literal["auto", "manual"] = "auto"
    embedding_model: Literal["jinaai/jina-embeddings-v3", "qwen3-0.6B"] = "jinaai/jina-embeddings-v3"
    manual_chunk_size: int = 1000
    manual_chunk_overlap: int = 200
    auto_token_threshold: int = 7000

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
    indexing_mode: str = "auto"
    embedding_model: str = "jinaai/jina-embeddings-v3"
    is_embedded: bool = True
    
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
            "file_size": str(self.file_size),
            "indexing_mode": str(self.indexing_mode),
            "embedding_model": str(self.embedding_model),
            "is_embedded": str(self.is_embedded)
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
        auto_token_threshold: int = 7000,
    ):
        print(f"🔧 Initializing Complete KB Service...")

        # Initialize components
        self.doc_converter = DocumentConverter()
        self.auto_token_threshold = auto_token_threshold
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Configuration
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize ChromaDB
        os.makedirs(vector_db_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=vector_db_path)
        self.collection = self._get_or_create_collection()
        
        # Initialize embedding models
        self.embedding_models = {}
        self._load_embedding_model(embedding_model_name)
        
        print(f"✅ Complete KB Service initialized successfully!")

    def _load_embedding_model(self, model_name: str):
        """Load embedding model by name"""
        if model_name in self.embedding_models:
            return self.embedding_models[model_name]
        
        print(f"📥 Loading embedding model: {model_name}")
        try:
            if model_name == "jinaai/jina-embeddings-v3":
                model = SentenceTransformer(model_name, trust_remote_code=True)
                model.max_seq_length = 8192
            elif model_name == "qwen3-0.6B":
                model = SentenceTransformer(model_name, trust_remote_code=True)
                model.max_seq_length = 8192
            else:
                raise ValueError(f"Unsupported embedding model: {model_name}")
                
            self.embedding_models[model_name] = model
            return model
        except Exception as e:
            print(f"❌ Failed to load model {model_name}: {str(e)}")
            print("🔄 Using fallback model...")
            fallback_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedding_models[model_name] = fallback_model
            return fallback_model

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

    def should_chunk_auto(self, content: str) -> bool:
        """Determine if content needs chunking in auto mode"""
        token_count = self.count_tokens(content)
        return token_count > self.auto_token_threshold

    def create_chunks_auto(
        self, content: str, metadata: Dict[str, Any], embedding_model: str
    ) -> List[Dict[str, Any]]:
        """Create chunks in auto mode based on token threshold"""
        timestamp = datetime.datetime.now().isoformat()
        token_count = self.count_tokens(content)
        
        # Check if content is below threshold
        if token_count <= self.auto_token_threshold:
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
                        indexing_mode="auto",
                        embedding_model=embedding_model,
                        is_embedded=False,  # Skip embedding for small content
                    ).to_dict(),
                }
            ]

        # Content is above threshold, chunk and embed
        return self._create_chunks_with_embedding(content, metadata, embedding_model, "auto")

    def create_chunks_manual(
        self, content: str, metadata: Dict[str, Any], chunk_size: int, chunk_overlap: int, embedding_model: str
    ) -> List[Dict[str, Any]]:
        """Create chunks in manual mode by character size"""
        return self._create_chunks_with_embedding(
            content, metadata, embedding_model, "manual", chunk_size, chunk_overlap
        )

    def _create_chunks_with_embedding(
        self, content: str, metadata: Dict[str, Any], embedding_model: str, 
        indexing_mode: str, chunk_size: int = None, chunk_overlap: int = None
    ) -> List[Dict[str, Any]]:
        """Create chunks with embedding (internal method)"""
        timestamp = datetime.datetime.now().isoformat()
        
        # Use provided chunk size/overlap or defaults
        if chunk_size is None:
            chunk_size = self.chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.chunk_overlap

        # Split into sentences
        sentences = self._split_into_sentences(content)
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            if current_tokens + sentence_tokens > chunk_size and current_chunk:
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
                            indexing_mode=indexing_mode,
                            embedding_model=embedding_model,
                            is_embedded=True,
                        ).to_dict(),
                    }
                )

                # Handle overlap
                overlap_content = self._get_overlap_content(
                    current_chunk, chunk_overlap
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
                        indexing_mode=indexing_mode,
                        embedding_model=embedding_model,
                        is_embedded=True,
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

    def generate_embeddings(self, chunks: List[Dict[str, Any]], embedding_model: str) -> List[Dict[str, Any]]:
        """Generate embeddings using specified model"""
        # Filter chunks that need embedding
        chunks_to_embed = [chunk for chunk in chunks if chunk["metadata"].get("is_embedded", "True") == "True"]
        
        if not chunks_to_embed:
            return chunks  # No chunks need embedding
        
        # Load model if not already loaded
        model = self._load_embedding_model(embedding_model)
        
        contents = [chunk["content"] for chunk in chunks_to_embed]
        embeddings = model.encode(contents, convert_to_numpy=True)

        # Update chunks with embeddings
        embed_index = 0
        for i, chunk in enumerate(chunks):
            if chunk["metadata"].get("is_embedded", "True") == "True":
                chunk["embedding"] = embeddings[embed_index].tolist()
                embed_index += 1

        return chunks

    def store_in_vector_db(self, chunks_with_embeddings: List[Dict[str, Any]]) -> bool:
        """Store in ChromaDB"""
        try:
            # Separate embedded and non-embedded chunks
            embedded_chunks = []
            non_embedded_chunks = []
            
            for chunk in chunks_with_embeddings:
                if chunk["metadata"].get("is_embedded", "True") == "True" and "embedding" in chunk:
                    embedded_chunks.append(chunk)
                else:
                    non_embedded_chunks.append(chunk)
            
            # Store embedded chunks in vector DB
            if embedded_chunks:
                ids = [chunk["metadata"]["chunk_id"] for chunk in embedded_chunks]
                documents = [chunk["content"] for chunk in embedded_chunks]
                embeddings = [chunk["embedding"] for chunk in embedded_chunks]

                # Ensure all metadata values are strings and non-None
                metadatas = []
                for chunk in embedded_chunks:
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
            
            # Store non-embedded chunks as raw content (no embeddings)
            if non_embedded_chunks:
                ids = [chunk["metadata"]["chunk_id"] for chunk in non_embedded_chunks]
                documents = [chunk["content"] for chunk in non_embedded_chunks]
                
                # Create zero embeddings for non-embedded chunks
                embedding_dim = 768  # Default dimension, will be adjusted by model
                if embedded_chunks and "embedding" in embedded_chunks[0]:
                    embedding_dim = len(embedded_chunks[0]["embedding"])
                
                zero_embeddings = [[0.0] * embedding_dim for _ in non_embedded_chunks]

                metadatas = []
                for chunk in non_embedded_chunks:
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
                    ids=ids, documents=documents, embeddings=zero_embeddings, metadatas=metadatas
                )

            return True
        except Exception as e:
            print(f"Error storing in vector DB: {str(e)}")
            return False

    def process_document(
        self, file_path: str, indexing_config: IndexingConfig, replace_existing: bool = True
    ) -> Dict[str, Any]:
        """Complete pipeline with indexing mode support"""
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

            # Step 3: Process document based on indexing mode
            token_count = self.count_tokens(content)
            
            if indexing_config.mode == "auto":
                print(f"🤖 Auto mode: {token_count} tokens (threshold: {self.auto_token_threshold})")
                needs_chunking = self.should_chunk_auto(content)
                chunks = self.create_chunks_auto(content, metadata, indexing_config.embedding_model)
            else:  # manual mode
                print(f"🔧 Manual mode: chunking by {indexing_config.manual_chunk_size} chars")
                needs_chunking = True
                chunks = self.create_chunks_manual(
                    content, metadata, 
                    indexing_config.manual_chunk_size, 
                    indexing_config.manual_chunk_overlap,
                    indexing_config.embedding_model
                )

            print(f"📝 Created {len(chunks)} chunks")

            # Step 4: Generate embeddings (only for chunks that need it)
            embedded_chunks = [chunk for chunk in chunks if chunk["metadata"].get("is_embedded", "True") == "True"]
            if embedded_chunks:
                print(f"🧮 Generating embeddings for {len(embedded_chunks)} chunks...")
                chunks_with_embeddings = self.generate_embeddings(chunks, indexing_config.embedding_model)
            else:
                print("⏭️ Skipping embeddings (content below threshold)")
                chunks_with_embeddings = chunks

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
                "indexing_mode": indexing_config.mode,
                "embedding_model": indexing_config.embedding_model,
                "message": f"Successfully {action} '{filename}' with {len(chunks)} chunks using {indexing_config.mode} mode and {indexing_config.embedding_model}"
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
            # Use the first available embedding model for querying
            if not self.embedding_models:
                return {"success": False, "error": "No embedding models available", "query": query}
            
            model_name = list(self.embedding_models.keys())[0]
            model = self.embedding_models[model_name]
            
            query_embedding = model.encode([query])
            
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
                        "indexing_mode": metadata.get("indexing_mode", "auto"),
                        "embedding_model": metadata.get("embedding_model", "unknown"),
                        "embedded_chunks": 0,
                        "raw_chunks": 0,
                    }
                doc_info[source]["chunks"] += 1
                
                # Count embedded vs raw chunks
                if metadata.get("is_embedded", "True") == "True":
                    doc_info[source]["embedded_chunks"] += 1
                else:
                    doc_info[source]["raw_chunks"] += 1

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
async def upload_document(
    file: UploadFile = File(...),
    indexing_mode: str = Query("auto", description="Indexing mode: 'auto' or 'manual'"),
    embedding_model: str = Query("jinaai/jina-embeddings-v3", description="Embedding model: 'jinaai/jina-embeddings-v3' or 'qwen3-0.6B'"),
    manual_chunk_size: int = Query(1000, description="Manual chunk size (characters)"),
    manual_chunk_overlap: int = Query(200, description="Manual chunk overlap (characters)"),
    auto_token_threshold: int = Query(7000, description="Auto mode token threshold"),
):
    """Upload and process document via API with indexing mode selection"""
    try:
        # Validate file type
        allowed_extensions = {".pdf", ".docx", ".doc", ".txt", ".md"}
        file_extension = os.path.splitext(file.filename)[1].lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type: {file_extension}"
            )

        # Validate indexing mode
        if indexing_mode not in ["auto", "manual"]:
            raise HTTPException(
                status_code=400, detail="Indexing mode must be 'auto' or 'manual'"
            )

        # Validate embedding model
        if embedding_model not in ["jinaai/jina-embeddings-v3", "qwen3-0.6B"]:
            raise HTTPException(
                status_code=400, detail="Embedding model must be 'jinaai/jina-embeddings-v3' or 'qwen3-0.6B'"
            )

        # Create indexing config
        indexing_config = IndexingConfig(
            mode=indexing_mode,
            embedding_model=embedding_model,
            manual_chunk_size=manual_chunk_size,
            manual_chunk_overlap=manual_chunk_overlap,
            auto_token_threshold=auto_token_threshold
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
            result = kb_service.process_document(temp_file_path, indexing_config)
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