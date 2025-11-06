#!/usr/bin/env python3
"""
rag_indexer.py

End-to-end, **text-only** RAG embedder with:
- S3 download (public or signed URLs) with graceful fallback to local files
- Robust text extraction for PDF / TXT / CSV / XLS / XLSX / JSON (text fields only)
- High-quality hierarchical chunking (sections → paragraphs → sentences) with overlap
- OpenAI embeddings (configurable model) with API key sourcing from Streamlit secrets or .env
- In-RAM vector index (NumPy) with cosine similarity + optional MongoDB persistence
- Streamlit-friendly logging (prints) and zero-crash failure path if no source found

Usage (Streamlit or CLI):
    from rag_indexer import RAGIndexer

    idx = RAGIndexer(
        s3_urls=[ ... ],                 # optional list of S3 object URLs (public/signed)
        local_paths=["/path/to/folder"], # optional local folder(s) or file paths
        mongo_uri=None,                  # optional: "mongodb+srv://..."
        mongo_db="ragdb",
        mongo_coll="documents",
        embed_model="text-embedding-3-large",
        max_tokens=900,      # chunk target size (approx. tokens, char-heuristic)
        overlap=150,         # chunk overlap (approx. tokens, char-heuristic)
        min_chunk_chars=280, # ensure chunks are not too tiny
    )
    idx.build()  # downloads (if possible), loads local fallback, extracts, chunks, embeds, in-RAM index

    # Query in-process (example)
    results = idx.query("What is the relocation entitlement for M3?", top_k=5)
    for r in results:
        print(r["score"], r["metadata"]["source"], r["text"][:120], "…")

Notes:
- Strictly ignores images; only textual content from supported file types is used.
- If neither S3 nor local sources are available/valid, exits gracefully with a console message.
"""

from __future__ import annotations
import os
import io
import re
import json
import math
import gzip
import time
import hashlib
import logging
import mimetypes
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterable, Union
from pathlib import Path

# Optional Streamlit compatibility
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None  # noqa: F401

# Env loader
from dotenv import load_dotenv
load_dotenv("./.env")

# S3 and Mongo (optional)
try:
    import boto3  # type: ignore
except Exception:
    boto3 = None  # noqa: F401

try:
    from pymongo import MongoClient  # type: ignore
except Exception:
    MongoClient = None  # noqa: F401

# PDFs
# PDFs — with full debug logging
# PDFs — with full debug logging
def _local_log(msg: str):
    """Local logger used ONLY for early import debugging (before _print exists)."""
    try:
        print(msg)
    except Exception:
        pass

try:
    import pypdf
    from pypdf import PdfReader
    _local_log(f"[DEBUG] ✅ pypdf imported successfully — version: {pypdf.__version__}")
except Exception as e:
    PdfReader = None
    _local_log("[DEBUG] ❌ Failed to import pypdf / PdfReader")
    _local_log(f"[DEBUG] Import error details: {e}")



# Excel/CSV
import csv
import pandas as pd

# Embeddings
import numpy as np

# OpenAI client (new SDK) with fallback to legacy
_OPENAI_CLIENT = None
def query(self, q: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if self.vectors is None or not self.texts:
        return []

    api = _get_api_key()
    if "error" in api:
        _print(f"[WARN] {api['error']}")
        return []
    api_key = api["key"]

    # Use the SAME embedding logic as batch
    qvec = embed_texts_openai([q], self.cfg.embed_model, api_key)[0]

    sims = np.dot(self.vectors, qvec) / (
        np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(qvec) + 1e-8
    )
    top_idx = np.argsort(-sims)[:top_k]

    results = []
    for i in top_idx:
        results.append({
            "index": int(i),
            "score": float(sims[i]),
            "text": self.texts[i],
            "metadata": self.metadatas[i],
        })
    return results

# -----------------------
# Helpers: auth & logging
# -----------------------

def _get_api_key() -> Dict[str, Any]:
    stage = "apikey"
    api_key = None

    # Streamlit secrets first (if available)
    if st and hasattr(st, "secrets"):
        api_key = st.secrets.get("OPENAI_API_KEY")

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return {"error": "Missing OpenAI API key (check Streamlit secrets or .env)", "stage": stage}

    return {"key": api_key, "stage": stage}

def _get_s3_creds() -> Optional[Dict[str, str]]:
    # Accept either Streamlit secrets or env
    def _pick(key):
        if st and hasattr(st, "secrets"):
            val = st.secrets.get(key)
            if val:
                return val
        return os.getenv(key)

    access_key = _pick("AWS_ACCESS_KEY_ID")
    secret_key = _pick("AWS_SECRET_ACCESS_KEY")
    region     = _pick("AWS_DEFAULT_REGION") or _pick("AWS_REGION")

    if access_key and secret_key:
        return {"aws_access_key_id": access_key, "aws_secret_access_key": secret_key, "region_name": region or "ap-south-1"}
    return None

def _get_mongo_uri() -> Optional[str]:
    if st and hasattr(st, "secrets"):
        val = st.secrets.get("MONGODB_URI")
        if val:
            return val
    return os.getenv("MONGODB_URI")

def _print(msg: str):
    # Streamlit-friendly print
    try:
        if st:
            st.write(msg)
        else:
            print(msg)
    except Exception:
        print(msg)

# -----------------------
# Text extraction (strictly text)
# -----------------------

def _read_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_text(encoding="latin-1", errors="ignore")

def _read_json_text(path: Path) -> str:
    # Extract only textual fields
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return ""
    texts: List[str] = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str):
                    texts.append(str(k))
                walk(v)
        elif isinstance(x, list):
            for i in x:
                walk(i)
        else:
            # Only textual scalars
            if isinstance(x, str):
                texts.append(x)

    walk(data)
    return "\n".join(t for t in texts if t and t.strip())

def _read_pdf_text(path: Path) -> str:
    if PdfReader is None:
        _print("pypdf not installed; skipping PDF: " + str(path))
        return ""
    try:
        reader = PdfReader(str(path))
        pages_text = []
        for i, page in enumerate(reader.pages):
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                pages_text.append(f"[[PAGE {i+1}]]\n{t}")
        return "\n\n".join(pages_text)
    except Exception as e:
        _print(f"PDF read error for {path.name}: {e}")
        return ""

def _read_csv_text(path: Path) -> str:
    try:
        df = pd.read_csv(path, dtype=str, low_memory=False)
    except Exception:
        # Fallback: python csv
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""
    df = df.fillna("")
    # Only textual join
    return "\n".join(
        " | ".join(str(x) for x in row if isinstance(x, str) and x.strip())
        for row in df.values.tolist()
    )

def _read_excel_text(path: Path) -> str:
    try:
        xls = pd.ExcelFile(path)
    except Exception:
        return ""
    texts = []
    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet, dtype=str)
            df = df.fillna("")
            txt = "\n".join(
                " | ".join(str(x) for x in row if isinstance(x, str) and x.strip())
                for row in df.values.tolist()
            )
            if txt.strip():
                texts.append(f"[[SHEET {sheet}]]\n{txt}")
        except Exception:
            continue
    return "\n\n".join(texts)

def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in [".txt", ".md"]:
        return _read_txt(path)
    if suffix in [".json"]:
        return _read_json_text(path)
    if suffix in [".pdf"]:
        return _read_pdf_text(path)
    if suffix in [".csv"]:
        return _read_csv_text(path)
    if suffix in [".xls", ".xlsx"]:
        return _read_excel_text(path)
    # unknown: try text
    try:
        return _read_txt(path)
    except Exception:
        return ""

# -----------------------
# Chunking (hierarchical + overlap)
# -----------------------

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z(0-9“"])')

def _rough_token_len(s: str) -> int:
    # Heuristic: 1 token ≈ 4 chars (English-ish)
    return max(1, int(len(s) / 4))

def split_into_sections(text: str) -> List[str]:
    # Split on page/section markers, big headings, or multiple newlines
    # Preserve structure for better locality
    parts = re.split(r"(?:\n\s*\[\[PAGE .*?\]\]\s*|\n{2,})", text)
    return [p.strip() for p in parts if p and p.strip()]

def split_paragraphs(section: str) -> List[str]:
    paras = re.split(r"\n{1,}", section)
    return [p.strip() for p in paras if p.strip()]

def split_sentences(paragraph: str) -> List[str]:
    # quick sentence split
    chunks = _SENT_SPLIT.split(paragraph.strip())
    return [c.strip() for c in chunks if c.strip()]

def smart_chunk(
    text: str,
    max_tokens: int = 900,
    overlap: int = 150,
    min_chunk_chars: int = 280
) -> List[str]:
    """
    Hierarchical splitting to produce semantically coherent chunks:
    sections -> paragraphs -> sentences, then pack into ~max_tokens with overlap.
    """
    if not text or not text.strip():
        return []

    sections = split_into_sections(text)
    out: List[str] = []

    for sec in sections:
        paras = split_paragraphs(sec)
        sentences: List[str] = []
        for p in paras:
            # keep large bullet blocks together
            if len(p) > 1000 and ("•" in p or "*" in p or "-" in p):
                sentences.append(p)
            else:
                sentences.extend(split_sentences(p) or [p])

        # pack sentences into windows
        curr: List[str] = []
        curr_tok = 0
        for s in sentences:
            stoks = _rough_token_len(s)
            if curr and (curr_tok + stoks > max_tokens):
                chunk = " ".join(curr).strip()
                if len(chunk) >= min_chunk_chars:
                    out.append(chunk)
                else:
                    # try to merge with next if too small later
                    if out and len(out[-1]) < min_chunk_chars:
                        out[-1] = (out[-1] + " " + chunk).strip()
                    else:
                        out.append(chunk)
                # overlap: carry tail of previous chunk
                if overlap > 0 and out:
                    tail = out[-1].split()[-overlap*4:]  # approx words
                    curr = [" ".join(tail)] if tail else []
                    curr_tok = _rough_token_len(" ".join(curr)) if curr else 0
                else:
                    curr = []
                    curr_tok = 0
            curr.append(s)
            curr_tok += stoks

        if curr:
            chunk = " ".join(curr).strip()
            if chunk:
                out.append(chunk)

    # Final smoothing for very small trailing chunks
    smoothed: List[str] = []
    for ch in out:
        if smoothed and len(ch) < min_chunk_chars and len(smoothed[-1]) < (max_tokens*4)//3:
            smoothed[-1] = (smoothed[-1] + "\n" + ch).strip()
        else:
            smoothed.append(ch)
    return smoothed

# -----------------------
# Embedding + in-RAM index
# -----------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

def embed_texts_openai(
    texts: List[str],
    model: str,
    api_key: str,
    batch_size: int = 128
) -> np.ndarray:
    """
    Robust, batched embedding function with safe fallbacks:
    1. OpenAI new client
    2. OpenAI legacy client
    3. VoyageAI (free-tier compatible)
    4. Local BGE encoder (no API key required)
    """

    def batched(seq, batch_size):
        for i in range(0, len(seq), batch_size):
            yield seq[i:i + batch_size]

    # -----------------------------------------
    # 1) Try OpenAI NEW client
    # -----------------------------------------
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        vectors = []
        for batch in batched(texts, batch_size):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                vecs = [np.array(r.embedding, dtype=np.float32) for r in resp.data]
                vectors.extend(vecs)
            except Exception as e:
                _print(f"[OpenAI-new/batch] Failed: {e}")
                raise e

        return np.vstack(vectors)

    except Exception as e:
        _print(f"[OpenAI-new] Failed → {e}")

    # -----------------------------------------
    # 2) Try OpenAI LEGACY client
    # -----------------------------------------
    try:
        import openai
        openai.api_key = api_key

        vectors = []
        for batch in batched(texts, batch_size):
            resp = openai.Embedding.create(model=model, input=batch)
            vecs = [np.array(v["embedding"], dtype=np.float32) for v in resp["data"]]
            vectors.extend(vecs)

        return np.vstack(vectors)

    except Exception as e:
        _print(f"[OpenAI-legacy] Failed → {e}")

    # -----------------------------------------
    # 3) VoyageAI fallback (free-tier friendly)
    # -----------------------------------------
    try:
        _print("[Fallback] Using VoyageAI: voyage-2-lite")

        import voyageai
        voyage_key = os.getenv("VOYAGE_API_KEY")
        if not voyage_key:
            raise RuntimeError("Missing VOYAGE_API_KEY in .env")

        vo = voyageai.Client(api_key=voyage_key)

        vectors = []
        for batch in batched(texts, batch_size):
            resp = vo.embed(model="voyage-2-lite", input=batch)
            vecs = [np.array(v, dtype=np.float32) for v in resp.embeddings]
            vectors.extend(vecs)

        return np.vstack(vectors)

    except Exception as e:
        _print(f"[VoyageAI] Failed → {e}")

    # -----------------------------------------
    # 4) Local BGE fallback (NO API NEEDED)
    # -----------------------------------------
    try:
        _print("[Fallback] Using local BGE model: BAAI/bge-base-en-v1.5")
        from sentence_transformers import SentenceTransformer
        model_local = SentenceTransformer("BAAI/bge-base-en-v1.5")

        vecs = model_local.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return vecs.astype(np.float32)

    except Exception as e:
        _print(f"[Local BGE] Failed → {e}")

    # -----------------------------------------
    # Nothing worked
    # -----------------------------------------
    raise RuntimeError("❌ All embedding backends failed.")

# -----------------------
# S3 fetch (graceful)
# -----------------------

def download_s3_objects(urls: Sequence[str], dest_dir: Path) -> List[Path]:
    """
    Download using boto3 if creds exist; otherwise try direct HTTP GET (for presigned/public URLs).
    If both fail, skip that URL. Returns list of downloaded paths.
    """
    downloaded: List[Path] = []
    dest_dir.mkdir(parents=True, exist_ok=True)

    sess = None
    if boto3:
        creds = _get_s3_creds()
        if creds:
            try:
                sess = boto3.session.Session(
                    aws_access_key_id=creds["aws_access_key_id"],
                    aws_secret_access_key=creds["aws_secret_access_key"],
                    region_name=creds["region_name"],
                )
            except Exception:
                sess = None

    s3 = sess.client("s3") if sess else None

    import urllib.parse, urllib.request

    for u in urls:
        try:
            parsed = urllib.parse.urlparse(u)
            # Try boto3 path (s3://bucket/key or https presigned to s3 domain)
            if s3 and (parsed.scheme == "s3"):
                bucket = parsed.netloc
                key = parsed.path.lstrip("/")
                fname = Path(key).name or hashlib.md5(u.encode()).hexdigest()
                dest = dest_dir / fname
                s3.download_file(bucket, key, str(dest))
                downloaded.append(dest)
                _print(f"[S3] downloaded: {dest.name}")
                continue

            # HTTP(S) fallback (works for presigned/public)
            fname = Path(parsed.path).name or hashlib.md5(u.encode()).hexdigest()
            dest = dest_dir / fname
            urllib.request.urlretrieve(u, str(dest))
            downloaded.append(dest)
            _print(f"[HTTP] downloaded: {dest.name}")
        except Exception as e:
            _print(f"[WARN] Could not fetch {u}: {e}")
            continue
    return downloaded

# -----------------------
# Mongo persistence (optional)
# -----------------------

def persist_to_mongo(
    docs: List[Dict[str, Any]],
    embeddings: np.ndarray,
    uri: str,
    db: str,
    coll: str
):
    if MongoClient is None:
        _print("[Mongo] pymongo not installed; skipping persistence.")
        return
    try:
        client = MongoClient(uri)
        collection = client[db][coll]
        payload = []
        for i, d in enumerate(docs):
            vec = embeddings[i].astype(float).tolist()
            payload.append({
                "text": d["text"],
                "metadata": d.get("metadata", {}),
                "embedding": vec,
                "ts": int(time.time()),
            })
        if payload:
            collection.insert_many(payload)
            _print(f"[Mongo] Inserted {len(payload)} docs into {db}.{coll}")
    except Exception as e:
        _print(f"[Mongo] Error: {e}")

# -----------------------
# Main class
# -----------------------

@dataclass
class RAGIndexerConfig:
    s3_urls: Optional[List[str]] = None
    local_paths: Optional[List[Union[str, Path]]] = None
    workdir: Union[str, Path] = "rag_work"
    embed_model: str = "text-embedding-3-large"
    max_tokens: int = 900
    overlap: int = 150
    min_chunk_chars: int = 280
    mongo_uri: Optional[str] = None
    mongo_db: str = "ragdb"
    mongo_coll: str = "documents"

class RAGIndexer:
    def __init__(
        self,
        s3_urls: Optional[List[str]] = None,
        local_paths: Optional[List[Union[str, Path]]] = None,
        workdir: Union[str, Path] = "rag_work",
        embed_model: str = "text-embedding-3-large",
        max_tokens: int = 900,
        overlap: int = 150,
        min_chunk_chars: int = 280,
        mongo_uri: Optional[str] = None,
        mongo_db: str = "ragdb",
        mongo_coll: str = "documents",
    ):
        self.cfg = RAGIndexerConfig(
            s3_urls=s3_urls or [],
            local_paths=[Path(p) for p in (local_paths or [])],
            workdir=Path(workdir),
            embed_model=embed_model,
            max_tokens=max_tokens,
            overlap=overlap,
            min_chunk_chars=min_chunk_chars,
            mongo_uri=mongo_uri or _get_mongo_uri(),
            mongo_db=mongo_db,
            mongo_coll=mongo_coll,
        )
        self.cfg.workdir.mkdir(parents=True, exist_ok=True)
        self.download_dir = self.cfg.workdir / "downloads"
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # In-RAM index state
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None

    # ---------- orchestration ----------
    def build(self):
        # 1) Resolve sources
        files = self._resolve_sources()
        if not files:
            _print("[INFO] No valid sources available (S3 and local both absent/unreadable). Exiting gracefully.")
            return

        # 2) Extract text corpus
        docs: List[Tuple[str, Dict[str, Any]]] = []
        for p in files:
            t = load_text(p)
            if t and t.strip():
                docs.append((t, {"source": str(p)}))
                _print(f"[OK] Loaded text from {p.name} ({len(t)} chars)")
            else:
                _print(f"[SKIP] No extractable text in {p.name}")

        if not docs:
            _print("[INFO] No textual content extracted. Exiting gracefully.")
            return

        # 3) Chunk
        chunked_docs: List[Dict[str, Any]] = []
        for text, meta in docs:
            chunks = smart_chunk(
                text,
                max_tokens=self.cfg.max_tokens,
                overlap=self.cfg.overlap,
                min_chunk_chars=self.cfg.min_chunk_chars,
            )
            for i, ch in enumerate(chunks):
                chunked_docs.append({
                    "text": ch,
                    "metadata": {**meta, "chunk": i, "n_chunks_in_source": len(chunks)}
                })
        _print(f"[CHUNK] Produced {len(chunked_docs)} chunks")

        # 4) Embeddings
        api = _get_api_key()
        if "error" in api:
            _print(f"[ABORT] {api['error']}")
            return
        api_key = api["key"]

        texts = [d["text"] for d in chunked_docs]
        vecs = embed_texts_openai(texts, self.cfg.embed_model, api_key)
        self.texts = texts
        self.metadatas = [d["metadata"] for d in chunked_docs]
        self.vectors = vecs
        _print(f"[EMBED] Embedded {vecs.shape[0]} chunks, dim={vecs.shape[1]} (model={self.cfg.embed_model})")

        # 5) Optional Mongo persistence
        if self.cfg.mongo_uri:
            try:
                persist_to_mongo(chunked_docs, vecs, self.cfg.mongo_uri, self.cfg.mongo_db, self.cfg.mongo_coll)
            except Exception as e:
                _print(f"[Mongo] Skipped persistence due to error: {e}")

    def query(self, q: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.vectors is None or not self.texts:
            return []
        api = _get_api_key()
        if "error" in api:
            _print(f"[WARN] {api['error']}")
            return []
        api_key = api["key"]

        qvec = embed_texts_openai([q], self.cfg.embed_model, api_key)[0]
        sims = np.dot(self.vectors, qvec) / (np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(qvec) + 1e-8)
        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for i in top_idx:
            results.append({
                "index": int(i),
                "score": float(sims[i]),
                "text": self.texts[i],
                "metadata": self.metadatas[i],
            })
        return results

    # ---------- sources ----------
    def _resolve_sources(self) -> List[Path]:
        files: List[Path] = []

        # S3 first (if provided)
        if self.cfg.s3_urls:
            _print("[S3] Attempting download(s)…")
            files.extend(download_s3_objects(self.cfg.s3_urls, self.download_dir))

        # Local fallback: folders or files
        for p in self.cfg.local_paths:
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.is_file() and f.suffix.lower() in (".pdf", ".txt", ".md", ".json", ".csv", ".xls", ".xlsx"):
                        files.append(f)

        # Deduplicate by absolute path
        uniq = []
        seen = set()
        for f in files:
            k = str(f.resolve())
            if k not in seen:
                uniq.append(f)
                seen.add(k)
        return uniq

# -------------- CLI --------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3", nargs="*", default=None, help="S3 public/presigned URLs (s3:// or https)")
    parser.add_argument("--paths", nargs="*", default=None, help="Local files or folders")
    parser.add_argument("--workdir", default="rag_work")
    parser.add_argument("--model", default="text-embedding-3-large")
    parser.add_argument("--max_tokens", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--min_chunk_chars", type=int, default=280)
    parser.add_argument("--mongo_uri", default=None)
    parser.add_argument("--mongo_db", default="ragdb")
    parser.add_argument("--mongo_coll", default="documents")
    args = parser.parse_args()

    idx = RAGIndexer(
        s3_urls=args.s3,
        local_paths=args.paths,
        workdir=args.workdir,
        embed_model=args.model,
        max_tokens=args.max_tokens,
        overlap=args.overlap,
        min_chunk_chars=args.min_chunk_chars,
        mongo_uri=args.mongo_uri,
        mongo_db=args.mongo_db,
        mongo_coll=args.mongo_coll,
    )
    idx.build()
    _print("[DONE] RAG in-RAM index ready.")
