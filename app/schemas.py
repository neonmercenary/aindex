from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class AddChunkRequest(BaseModel):
    document_name: str
    page_number: int
    content: str