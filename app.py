# unified_kb_app.py
import os
import hashlib
import tempfile
import threading
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import uuid

import gradio as gr
from gradio.routes import mount_gradio_app
from gradio.blocks import Blocks
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from docling.document_converter import DocumentConverter
import chromadb
from sentence_transformers import SentenceTransformer
import tiktoken


# Pydantic models for API
class QueryRequest(BaseModel):
    query: str
    n_results: int = 5


class QueryResponse(BaseModel):
    success: bool
    query: str
    results: dict
    count: int
    error: str = None


@dataclass
class ChunkMetadata:
    source: str
    chunk_id: str
    document_hash: str  # Add document hash for tracking
    page_number: int = 0
    chunk_index: int = 0
    total_chunks: int = 0
    file_type: str = "unknown"
    upload_timestamp: str = ""  # Add timestamp

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary with string values for ChromaDB"""
        return {
            "source": str(self.source),
            "chunk_id": str(self.chunk_id),
            "document_hash": str(self.document_hash),
            "page_number": str(self.page_number),
            "chunk_index": str(self.chunk_index),
            "total_chunks": str(self.total_chunks),
            "file_type": str(self.file_type),
            "upload_timestamp": str(self.upload_timestamp),
        }


class EnhancedKBService:
    def __init__(
        self,
        embedding_model_name: str = "jinaai/jina-embeddings-v3",
        max_tokens: int = 8000,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        vector_db_path: str = "./chroma_db",
    ):
        print(f"🔧 Initializing Enhanced KB Service...")

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
        print(f"✅ Enhanced KB Service initialized successfully!")

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
        # Use filename + content hash for uniqueness
        combined = f"{filename}:{content[:1000]}"  # Use first 1000 chars for efficiency
        return hashlib.md5(combined.encode()).hexdigest()

    def check_document_exists(self, document_hash: str, filename: str) -> bool:
        """Check if document already exists in vector DB"""
        try:
            # Query by document hash or filename
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
            # Build where clause
            where_clause = {}
            if document_hash:
                where_clause["document_hash"] = document_hash
            elif filename:
                where_clause["source"] = filename
            else:
                return {"success": False, "error": "No identifier provided"}

            # Get existing chunks
            existing_chunks = self.collection.get(where=where_clause)
            chunks_to_delete = len(existing_chunks["ids"])

            if chunks_to_delete > 0:
                # Delete chunks
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

            # Calculate document hash for tracking
            doc_hash = self.calculate_document_hash(content, filename)

            metadata = {
                "source": filename,
                "file_path": file_path,
                "file_type": file_type,
                "document_hash": doc_hash,
                "title": getattr(result.document, "title", filename),
                "page_count": (
                    len(result.document.pages)
                    if hasattr(result.document, "pages")
                    else 1
                ),
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
        import datetime

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
                        page_number=metadata.get("page_count", 1),
                        chunk_index=0,
                        total_chunks=1,
                        file_type=metadata.get("file_type", "unknown"),
                        upload_timestamp=timestamp,
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
                            page_number=metadata.get("page_count", 1),
                            chunk_index=chunk_index,
                            total_chunks=0,
                            file_type=metadata.get("file_type", "unknown"),
                            upload_timestamp=timestamp,
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
                        page_number=metadata.get("page_count", 1),
                        chunk_index=chunk_index,
                        total_chunks=0,
                        file_type=metadata.get("file_type", "unknown"),
                        upload_timestamp=timestamp,
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

    def query_knowledge_base(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the knowledge base"""
        try:
            query_embedding = self.embedding_model.encode([query])

            results = self.collection.query(
                query_embeddings=query_embedding.tolist(), n_results=n_results
            )

            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results["documents"][0]) if results["documents"] else 0,
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
                    }
                doc_info[source]["chunks"] += 1

            return list(doc_info.values())
        except Exception as e:
            return []


# Initialize the enhanced KB service
print("🚀 Starting Enhanced Knowledge Base System with Document Replacement...")
kb_service = EnhancedKBService()


# Update the Gradio upload function
def upload_and_process(file):
    """Upload and process document via Gradio with replacement support"""
    if file is None:
        return "Please select a file to upload.", ""

    try:
        # Always replace existing documents
        result = kb_service.process_document(file.name, replace_existing=True)

        if result["success"]:
            stats = kb_service.get_collection_stats()

            # Create status message
            action_emoji = "🔄" if result["action"] == "replaced" else "✅"
            action_text = "Replaced" if result["action"] == "replaced" else "Added"

            status_msg = f"""
            {action_emoji} **{action_text}!** 
            - File: {result['file_name']}
            - Action: {result['action'].title()}
            - Tokens: {result['token_count']:,}
            - New chunks: {result['chunks_created']}
            {f"- Deleted old chunks: {result['chunks_deleted']}" if result.get('chunks_deleted', 0) > 0 else ""}
            - Chunking needed: {result['needs_chunking']}
            - Total chunks in DB: {stats.get('total_chunks', 0)}
            - Unique documents: {stats.get('unique_documents', 0)}
            """
            return status_msg, ""
        else:
            return f"❌ **Error:** {result['error']}", ""

    except Exception as e:
        return f"❌ **Error:** {str(e)}", ""


def get_database_stats():
    """Get enhanced database statistics"""
    try:
        stats = kb_service.get_collection_stats()
        docs = kb_service.list_documents()

        doc_list = ""
        if docs:
            doc_list = "\n\n**Documents:**\n" + "\n".join(
                [
                    f"- {doc['filename']} ({doc['chunks']} chunks, {doc['file_type']})"
                    for doc in docs[:10]  # Show first 10
                ]
            )
            if len(docs) > 10:
                doc_list += f"\n... and {len(docs) - 10} more"

        return f"""📊 **Database Stats:**
- Total chunks: {stats.get('total_chunks', 0)}
- Unique documents: {stats.get('unique_documents', 0)}
{doc_list}
"""
    except Exception as e:
        return f"❌ **Error:** {str(e)}"


# Create FastAPI app
app = FastAPI(
    title="Knowledge Base API",
    description="Unified Knowledge Base with Gradio UI and REST APIs",
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


# FastAPI Routes
@app.post("/api/upload")
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
            return result
        finally:
            # Clean up temp file
            os.unlink(temp_file_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the knowledge base via API"""
    try:
        result = kb_service.query_knowledge_base(request.query, request.n_results)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Get collection statistics via API"""
    return kb_service.get_collection_stats()


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Knowledge Base API"}


# Gradio Interface Functions
def upload_and_process(file):
    """Upload and process document via Gradio"""
    if file is None:
        return "Please select a file to upload.", ""

    try:
        result = kb_service.process_document(file.name)

        if result["success"]:
            stats = kb_service.get_collection_stats()
            status_msg = f"""
            ✅ **Success!** 
            - File: {result['file_name']}
            - Tokens: {result['token_count']:,}
            - Chunks: {result['chunks_created']}
            - Chunking needed: {result['needs_chunking']}
            - Total documents in DB: {stats.get('total_documents', 0)}
            """
            return status_msg, ""
        else:
            return f"❌ **Error:** {result['error']}", ""

    except Exception as e:
        return f"❌ **Error:** {str(e)}", ""


def query_documents(query, n_results):
    """Query the knowledge base via Gradio - Fixed version"""
    if not query.strip():
        return "Please enter a query.", []

    try:
        result = kb_service.query_knowledge_base(query, n_results)

        if result["success"] and result["count"] > 0:
            # Format results
            formatted_results = []
            documents = result["results"]["documents"][0]
            metadatas = result["results"]["metadatas"][0]
            distances = result["results"]["distances"][0]

            for i, (doc, metadata, distance) in enumerate(
                zip(documents, metadatas, distances)
            ):
                # Fix: Convert all values to strings with proper handling
                chunk_index = int(metadata.get("chunk_index", 0))
                total_chunks = int(metadata.get("total_chunks", 1))

                formatted_results.append(
                    [
                        i + 1,
                        str(metadata.get("source", "Unknown")),
                        f"{chunk_index + 1}/{total_chunks}",  # Ensure proper int conversion
                        f"{1 - float(distance):.3f}",  # Ensure float conversion
                        doc[:200] + "..." if len(doc) > 200 else doc,
                    ]
                )

            return f"Found {result['count']} relevant results:", formatted_results
        else:
            return "No relevant documents found.", []

    except Exception as e:
        return f"❌ **Error:** {str(e)}", []


# Create Gradio interface
with gr.Blocks(
    title="Knowledge Base System",
    css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .main-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
    """,
) as demo:

    gr.HTML(
        """
        <div class="main-header">
            <h1>🧠 Unified Knowledge Base System</h1>
            <p>Upload documents, query with natural language, and access via REST APIs</p>
        </div>
    """
    )

    with gr.Tabs():
        with gr.Tab("📤 Upload Documents"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(
                        label="📄 Select Document",
                        file_types=[".pdf", ".docx", ".doc", ".txt", ".md"],
                        file_count="single",
                    )
                    upload_btn = gr.Button(
                        "🚀 Upload & Process", variant="primary", size="lg"
                    )

                with gr.Column(scale=2):
                    upload_status = gr.Markdown("Ready to upload documents...")

            upload_btn.click(
                upload_and_process,
                inputs=[file_input],
                outputs=[upload_status, gr.Textbox(visible=False)],
            )

        with gr.Tab("🔍 Query Knowledge Base"):
            with gr.Row():
                with gr.Column(scale=2):
                    query_input = gr.Textbox(
                        label="💬 Enter your question",
                        placeholder="What would you like to know?",
                        lines=3,
                    )

                with gr.Column(scale=1):
                    n_results = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="📊 Number of results",
                    )
                    query_btn = gr.Button("🔍 Search", variant="primary", size="lg")

            query_status = gr.Markdown("Enter a question to search...")

            results_table = gr.Dataframe(
                headers=["#", "Source", "Chunk", "Similarity", "Content"],
                datatype=["number", "str", "str", "str", "str"],
                label="🎯 Search Results",
                interactive=False,
                wrap=True,
            )

            query_btn.click(
                query_documents,
                inputs=[query_input, n_results],
                outputs=[query_status, results_table],
            )

        with gr.Tab("📊 Database Stats"):
            with gr.Row():
                with gr.Column():
                    stats_btn = gr.Button(
                        "📊 Refresh Stats", variant="secondary", size="lg"
                    )
                    stats_output = gr.Markdown("Click to view database statistics...")

                with gr.Column():
                    gr.Markdown(
                        """
                    ### 📈 System Information
                    - **Embedding Model**: Jina 8K Context
                    - **Vector DB**: ChromaDB
                    - **Max Tokens**: 8,000
                    - **Chunk Size**: 1,000 tokens
                    - **Overlap**: 200 tokens
                    """
                    )

            stats_btn.click(get_database_stats, outputs=[stats_output])

    gr.Markdown(
        """
    ---
    ## 🚀 REST API Endpoints
    
    **Base URL:** `http://localhost:7860`
    
    | Method | Endpoint | Description |
    |--------|----------|-------------|
    | POST | `/api/upload` | Upload and process documents |
    | POST | `/api/query` | Query the knowledge base |
    | GET | `/api/stats` | Get database statistics |
    | GET | `/api/health` | Health check |
    
    **Interactive API Documentation:** [http://localhost:7860/docs](http://localhost:7860/docs)
    
    ### Example Usage:
    ```
    # Upload document
    curl -X POST "http://localhost:7860/api/upload" -F "file=@document.pdf"
    
    # Query knowledge base
    curl -X POST "http://localhost:7860/api/query" \\
         -H "Content-Type: application/json" \\
         -d '{"query": "What is the main topic?", "n_results": 5}'
    ```
    
    **Built with:** Gradio + FastAPI + Docling + Jina Embeddings + ChromaDB
    """
    )

# Mount Gradio app on FastAPI
app = mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn

    print("🌟 Starting Unified Knowledge Base System on http://localhost:7860")
    print("📖 Gradio UI: http://localhost:7860")
    print("🔗 API Docs: http://localhost:7860/docs")

    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
