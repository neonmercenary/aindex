def split_text_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Splits text into chunks of a given character size with a specified overlap."""
    if not text:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        # Move forward by chunk_size minus the overlap to create the sliding window
        start += chunk_size - chunk_overlap
        
    return chunks