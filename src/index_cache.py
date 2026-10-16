import hashlib
import json
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent.parent / ".index_cache"

def make_cache_key(
    file_bytes, 
    model_name, 
    chunk_size, 
    overlap, 
    file_extension, 
    chunk_strategy
):
    settings = {
        "file_hash": hashlib.sha256(file_bytes).hexdigest(),
        "model": model_name,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "file_extension": file_extension.lower(),
        "chunk_strategy": chunk_strategy,
        "version": 3
    }

    encoded = json.dumps(settings, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def save_index(cache_key, chunks, vectors):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    path = CACHE_DIR / f"{cache_key}.npz"

    np.savez_compressed(
        path,
        chunks_json=np.array(json.dumps(chunks)),
        vectors=vectors
    )

def load_index(cache_key):
    path = CACHE_DIR / f"{cache_key}.npz"

    if not path.is_file():
        return None

    try:
        with np.load(path, allow_pickle=False) as saved:
            chunks = json.loads(saved["chunks_json"].item())
            vectors = saved["vectors"]

        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            return None

        if not np.isfinite(vectors).all():
            return None

        return chunks, vectors
    except (OSError, ValueError, KeyError, EOFError):
        return None
