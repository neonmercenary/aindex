import sqlite3
import sqlite_vec
import struct

DB_PATH = "rag_database.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    # Enable loading extensions
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    
    # Return rows as dictionaries for easier handling
    conn.row_factory = sqlite3.Row
    return conn

def serialize_vector(vector: list[float]) -> bytes:
    """Serializes a list of floats into raw bytes for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Relational table for storing text content and page numbers
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_name TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            content TEXT NOT NULL
        )
    """)
    
    # 2. Virtual table for vector similarity search (assuming 384 dimensions for all-MiniLM-L6-v2)
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            chunk_id INTEGER,
            embedding float[384]
        )
    """)
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully with sqlite-vec!")