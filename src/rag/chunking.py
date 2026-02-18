# rag/chunking.py
# -*- coding: utf-8 -*-
"""
Text chunking utilities for RAG pipeline.
"""
from dataclasses import dataclass
from typing import List

@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    chunk_id: str
    text: str
    meta: dict

def simple_chunk_text(
    doc_id: str,
    text: str,
    chunk_size: int = 900,
    overlap: int = 150
) -> List[Chunk]:
    """
    Split text into overlapping chunks.
    
    Args:
        doc_id: Document identifier
        text: Text to chunk
        chunk_size: Target chunk size in words
        overlap: Overlap size in words
    
    Returns:
        List of Chunk objects
    """
    words = text.split()
    chunks = []
    
    # Handle short text
    if len(words) <= chunk_size:
        return [Chunk(
            chunk_id=f"{doc_id}::chunk0",
            text=text,
            meta={"start": 0, "end": len(text)}
        )]
    
    start = 0
    chunk_idx = 0
    
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        
        chunks.append(Chunk(
            chunk_id=f"{doc_id}::chunk{chunk_idx}",
            text=chunk_text,
            meta={"start": start, "end": end}
        ))
        
        chunk_idx += 1
        start += (chunk_size - overlap)
    
    return chunks