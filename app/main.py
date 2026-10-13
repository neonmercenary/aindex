from fastapi import (
    FastAPI, HTTPException,
    UploadFile, File
)
from .settings import settings
from contextlib import asynccontextmanager
from .helpers import split_text_into_chunks
from fastembed import TextEmbedding
from .schemas import AddChunkRequest, QueryRequest
from .db import get_db_connection, serialize_vector, init_db
import pdfplumber
import io
import httpx
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the database on startup
    init_db()
    yield
    # Perform any cleanup if necessary on shutdown
    pass

app = FastAPI(title="Multimodal PDF RAG API", version="1.0", lifespan=lifespan)

# Initialize FastEmbed (runs on ONNX, zero PyTorch overhead)
embedding_model = TextEmbedding(model_name=settings.EMBEDDING_MODEL_NAME)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")

    

@app.post("/api/v1/chunks")
def add_chunk(data: AddChunkRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Insert in table
        cursor.execute(
            "INSERT INTO chunks (document_name, page_number, content) VALUES (?, ?, ?)",
            (data.document_name, data.page_number, data.content)
        )
        chunk_id = cursor.lastrowid
        
        # Generate embedding vector via fastembed (returns generator, take first item and convert to list)
        vector = list(embedding_model.embed([data.content]))[0].tolist()
        vector_bytes = serialize_vector(vector)
        
        # Insert into vector search table
        cursor.execute(
            "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, vector_bytes)
        )
        
        conn.commit()
        return {"status": "success", "chunk_id": chunk_id}
    
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.post("/api/v1/query")
def query_rag(data: QueryRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Convert user query to vector
    query_vector = list(embedding_model.embed([data.query]))[0].tolist()
    query_bytes = serialize_vector(query_vector)
    
    # Perform KNN vector search using sqlite-vec match operator with explicit k constraint
    sql = """
        SELECT 
            c.id, 
            c.document_name, 
            c.page_number, 
            c.content, 
            v.distance
        FROM vec_chunks v
        JOIN chunks c ON v.chunk_id = c.id
        WHERE v.embedding MATCH ? AND v.k = ?
        ORDER BY v.distance
    """
    
    cursor.execute(sql, (query_bytes, data.top_k))
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        results.append({
            "chunk_id": row["id"],
            "document": row["document_name"],
            "page": row["page_number"],
            "content": row["content"],
            "distance": row["distance"]
        })
        
    return {"query": data.query, "matches": results}



@app.post("/api/v1/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        file_bytes = file.file.read()
        conn = get_db_connection()
        cursor = conn.cursor()
        total_chunks = 0
        
        # Open PDF from memory bytes using pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                # Extract page  
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_chunk = split_text_into_chunks(page_text.strip())

                    for chunks_content in text_chunk:
                        cursor.execute(
                            "INSERT INTO chunks (document_name, page_number, content) VALUES (?, ?, ?)",
                            (file.filename, page_idx, chunks_content)
                        )
                        chunk_id = cursor.lastrowid
                    
                        # Generate embedding and insert into vec_chunks
                        vector = list(embedding_model.embed([chunks_content]))[0].tolist()
                        cursor.execute(
                            "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                            (chunk_id, serialize_vector(vector))
                        )
                        total_chunks += 1
                
                # Extract structured tables
                tables = page.extract_tables()
                for table in tables:
                    # Convert table rows to a readable layout string
                    table_lines = [" | ".join([str(cell or "").strip() for cell in row]) for row in table]
                    table_content = "[TABLE EXTRACT]\n" + "\n".join(table_lines)
                    
                    cursor.execute(
                        "INSERT INTO chunks (document_name, page_number, content) VALUES (?, ?, ?)",
                        (file.filename, page_idx, table_content)
                    )
                    chunk_id = cursor.lastrowid
                    
                    vector = list(embedding_model.embed([table_content]))[0].tolist()
                    cursor.execute(
                        "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                        (chunk_id, serialize_vector(vector))
                    )
                    total_chunks += 1
                    
        conn.commit()
        conn.close()
        return {
            "status": "success",
            "filename": file.filename,
            "total_chunks_indexed": total_chunks
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")




@app.post("/api/v1/generate")
def generate_rag_answer(data: QueryRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Embed query and search via sqlite-vec 
    query_vector = list(embedding_model.embed([data.query]))[0].tolist()
    query_bytes = serialize_vector(query_vector)
    
    sql = """
        SELECT 
            c.document_name, 
            c.page_number, 
            c.content, 
            v.distance
        FROM vec_chunks v
        JOIN chunks c ON v.chunk_id = c.id
        WHERE v.embedding MATCH ? AND v.k = ?
        ORDER BY v.distance
    """
    cursor.execute(sql, (query_bytes, data.top_k))
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return {"answer": "I couldn't find any relevant information in the uploaded documents.", "citations": []}
    
    # Build context string and collect citations
    context_blocks = []
    citations = set()
    for row in rows:
        context_blocks.append(f"[Document: {row['document_name']}, Page: {row['page_number']}]\n{row['content']}")
        citations.add(f"{row['document_name']} (Page {row['page_number']})")
        
    context_text = "\n\n".join(context_blocks)
    
    # Refine prompt based on model (Concise prompt for small models for speed, more detailed for larger models)
    prompt = f"""
        Answer the question based on the following context only.
        Indicate if the context doesn't contain enough information to answer the question.
        Context:    {context_text}
        Question:   {data.query}
    """


    # Call LLM
    try:
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma4_e4b_custom", 
                "prompt": prompt,
                "stream": False
            },
            timeout=120.0
        )
        
        # Print status and raw text to your FastAPI terminal to debug
        # print(f"Ollama Status Code: {response.status_code}")
        # print(f"Ollama Raw Response: {response.text}")
        
        result = response.json()
        answer_text = result.get("response", "Error generating response from local LLM.")

    except Exception as e:
        print(f"Ollama Connection Error: {str(e)}")
        answer_text = f"Failed to connect to local Ollama server: {str(e)}"


    return {
        "query": data.query,
        "answer": answer_text,
        "sources": list(citations)
    }