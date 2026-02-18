#!/usr/bin/env python3
"""
Step 0 — Prepare BM25 index for Run 3.

Your existing setup has:
  - output/step1/bm25_index.json   (JSON-serialized BM25)
  - output/step1/chunks.jsonl      (merged chunks)
  - output/step1/chunks_*.jsonl    (per-paper chunks)

Run 3 extraction needs:
  - A BM25 object that has .get_scores(tokenized_query)
  - A list of chunk dicts with keys: text, source_file, chunk_id

This script loads your existing data and creates a pickle index
compatible with Run 3, OR (if rank_bm25 is installed) rebuilds
a fresh BM25Okapi index from the merged chunks.
"""

import json
import os
import sys
import pickle
from pathlib import Path

INDEX_DIR = os.environ.get("KG_INDEX_DIR", "output/step1")


def load_chunks(index_dir):
    """Load chunks from the merged chunks.jsonl file."""
    chunks_path = os.path.join(index_dir, "chunks.jsonl")
    if not os.path.exists(chunks_path):
        print(f"❌ {chunks_path} not found")
        sys.exit(1)
    
    chunks = []
    with open(chunks_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Ensure required fields
                if "text" not in obj:
                    continue
                if "source_file" not in obj:
                    obj["source_file"] = obj.get("file", obj.get("paper", f"unknown_{i}"))
                if "chunk_id" not in obj:
                    obj["chunk_id"] = i
                chunks.append(obj)
            except json.JSONDecodeError:
                continue
    
    # Filter empty chunks
    chunks = [c for c in chunks if c["text"].strip()]
    return chunks


def build_bm25_from_chunks(chunks):
    """Build a BM25Okapi index from chunks."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("❌ rank_bm25 not installed. Install with: pip install rank-bm25")
        print("   Or: pip install rank-bm25 --break-system-packages")
        sys.exit(1)
    
    print(f"  Tokenizing {len(chunks)} chunks...")
    tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
    
    print(f"  Building BM25Okapi index...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    return bm25


def try_load_existing_json_index(index_dir):
    """Try to load the existing JSON BM25 index and check if it's usable."""
    json_path = os.path.join(index_dir, "bm25_index.json")
    if not os.path.exists(json_path):
        return None
    
    print(f"  Found existing {json_path} ({os.path.getsize(json_path)/1024/1024:.1f} MB)")
    
    try:
        with open(json_path) as f:
            data = json.load(f)
        
        # Check what format it is
        if isinstance(data, dict):
            print(f"  JSON index keys: {list(data.keys())[:10]}")
            # If it contains the BM25 parameters, we can reconstruct
            if "idf" in data or "doc_freqs" in data or "avgdl" in data:
                print("  → Looks like serialized BM25 parameters")
                return data
            elif "corpus" in data:
                print("  → Looks like a corpus index")
                return data
        elif isinstance(data, list):
            print(f"  → JSON is a list of {len(data)} items")
            return data
            
        print("  → Unknown format, will rebuild from chunks")
        return None
        
    except (json.JSONDecodeError, MemoryError) as e:
        print(f"  ⚠️ Could not load JSON index: {e}")
        return None


def main():
    print("=" * 60)
    print("  Step 0: Prepare BM25 Index for Run 3")
    print("=" * 60)
    
    # 1. Load chunks
    print(f"\n1. Loading chunks from {INDEX_DIR}...")
    chunks = load_chunks(INDEX_DIR)
    print(f"   ✅ {len(chunks)} non-empty chunks loaded")
    
    # Show sample
    if chunks:
        sample = chunks[0]
        print(f"   Sample chunk keys: {list(sample.keys())}")
        print(f"   Sample text (first 100 chars): {sample['text'][:100]}...")
        print(f"   Sample source_file: {sample.get('source_file', 'N/A')}")
    
    # 2. Check existing index
    print(f"\n2. Checking existing BM25 index...")
    existing = try_load_existing_json_index(INDEX_DIR)
    
    # 3. Build/rebuild BM25
    print(f"\n3. Building BM25Okapi index from chunks...")
    bm25 = build_bm25_from_chunks(chunks)
    
    # 4. Save as pickle (what Run 3 expects)
    pkl_path = os.path.join(INDEX_DIR, "bm25_index.pkl")
    print(f"\n4. Saving pickle index to {pkl_path}...")
    with open(pkl_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"   ✅ Saved ({os.path.getsize(pkl_path)/1024/1024:.1f} MB)")
    
    # 5. Verify
    print(f"\n5. Verification...")
    with open(pkl_path, "rb") as f:
        bm25_check = pickle.load(f)
    
    test_query = "chaotic seismic facies mass transport deposit".split()
    scores = bm25_check.get_scores(test_query)
    top_idx = scores.argsort()[-3:][::-1]
    
    print(f"   Test query: {' '.join(test_query)}")
    for rank, idx in enumerate(top_idx):
        print(f"   Top-{rank+1}: score={scores[idx]:.2f} | "
              f"file={chunks[idx].get('source_file','?')} | "
              f"text={chunks[idx]['text'][:80]}...")
    
    # 6. Summary
    sources = set(c.get("source_file", "unknown") for c in chunks)
    print(f"\n{'=' * 60}")
    print(f"  READY FOR RUN 3")
    print(f"{'=' * 60}")
    print(f"  Chunks:       {len(chunks)}")
    print(f"  Source papers: {len(sources)}")
    print(f"  BM25 pickle:  {pkl_path}")
    print(f"  Chunks JSONL:  {os.path.join(INDEX_DIR, 'chunks.jsonl')}")
    print(f"\n  Next: python run3_fixed/02_rag_extract_triples_v3.py")


if __name__ == "__main__":
    main()