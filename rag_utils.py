import json
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Tuple

import google.generativeai as genai
import numpy as np

BANGLA_CHAR_RE = re.compile(r"[\u0980-\u09FF]")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")


def normalize_bangla_text(text: str) -> str:
    """Normalize Bangla/romanized text extracted from PDFs."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\u200c", "").replace("\u200d", "")  # ZWNJ/ZWJ
    text = text.replace("\xa0", " ")
    text = re.sub(r"[•●▪◦·]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def paragraph_chunks(text: str, *, max_chars: int = 600, min_chars: int = 200) -> List[str]:
    """Split normalized text into roughly paragraph-sized chunks."""
    if not text:
        return []

    sentences = re.split(r"(?<=[।!?]|\n)\s+", text)
    out, buf = [], []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(" ".join(buf + [sentence])) > max_chars and buf:
            chunk = " ".join(buf).strip()
            if chunk:
                out.append(chunk)
            buf = [sentence]
        else:
            buf.append(sentence)

    if buf:
        chunk = " ".join(buf).strip()
        if chunk:
            out.append(chunk)

    # If chunks are very short, merge with previous chunk
    merged: List[str] = []
    for chunk in out:
        if merged and len(merged[-1]) < min_chars:
            merged[-1] = f"{merged[-1]} {chunk}".strip()
        else:
            merged.append(chunk)
    return merged


def ensure_dir(path: os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=2048)
def _cached_embedding(model: str, payload: str) -> Tuple[float, ...]:
    resp = genai.embed_content(model=model, content=payload or ".")
    embedding = resp.get("embedding") or resp.get("vector") or resp
    if isinstance(embedding, dict) and "embedding" in embedding:
        embedding = embedding["embedding"]
    if not isinstance(embedding, list):
        raise ValueError("Unexpected embedding response format")
    return tuple(float(x) for x in embedding)


def embed_texts(texts: Iterable[str], *, model: str = EMBED_MODEL) -> np.ndarray:
    """Return embeddings for each text using Gemini embedding API with simple caching."""
    vectors: List[Tuple[float, ...]] = []
    for text in texts:
        payload = normalize_bangla_text(text)
        vectors.append(_cached_embedding(model, payload or "."))
    return np.asarray(vectors, dtype=np.float32)


def contains_bangla(text: str) -> bool:
    return bool(BANGLA_CHAR_RE.search(text or ""))


def save_index(embeddings: np.ndarray, records: List[dict], *, out_dir: os.PathLike) -> Tuple[str, str]:
    ensure_dir(out_dir)
    embeddings_path = Path(out_dir) / "rag_embeddings.npy"
    metadata_path = Path(out_dir) / "rag_metadata.json"
    np.save(embeddings_path, embeddings.astype(np.float32))
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return str(embeddings_path), str(metadata_path)


def load_index(embeddings_path: str, metadata_path: str):
    if not (os.path.exists(embeddings_path) and os.path.exists(metadata_path)):
        return None
    embeddings = np.load(embeddings_path).astype(np.float32)
    with open(metadata_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    source_index = {}
    for idx, rec in enumerate(records):
        norm_text = normalize_bangla_text(rec.get("text", ""))
        rec["_norm_text"] = norm_text
        src = rec.get("source")
        if src:
            source_index.setdefault(src, []).append(idx)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    return {
        "embeddings": embeddings,
        "records": records,
        "norms": norms,
        "source_index": source_index,
    }


def cosine_top_k(index, query_vec: np.ndarray, k: int = 4):
    if not index:
        return []
    embeddings = index["embeddings"]
    norms = index["norms"]
    query_vec = query_vec.astype(np.float32)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return []
    sims = (embeddings @ query_vec) / (norms.flatten() * (query_norm + 1e-8))
    k = min(k, len(sims))
    if k <= 0:
        return []
    top_indices = np.argsort(-sims)[:k]
    out = []
    for idx in top_indices:
        record = index["records"][int(idx)]
        rec = {
            "index": int(idx),
            "score": float(sims[idx]),
            **record,
        }
        out.append(rec)
    return out
