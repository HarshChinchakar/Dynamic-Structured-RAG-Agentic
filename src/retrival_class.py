#!/usr/bin/env python3
"""
retriever.py

Retriever that works with in-RAM embeddings or an in-memory Chroma store.
- Top-K retrieval using cosine similarity
- Optional LLM-based re-ranking (with safe fallbacks)
- Returns structured results suitable to forward to another component

Usage example:

from retriever import Retriever
# If you have RAGIndexer instance 'idx' (with .texts, .metadatas, .vectors):
ret = Retriever(texts=idx.texts, metadatas=idx.metadatas, vectors=idx.vectors)
results = ret.retrieve("What is the relocation entitlement for M3?", top_k=15, rerank=True)
"""

from __future__ import annotations
import os
import json
import math
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path

# dotenv + optional Streamlit secrets
from dotenv import load_dotenv
load_dotenv("./.env")
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None

# numeric processing
import numpy as np

# Optional langchain / chroma imports (use if available)
try:
    # this import name may vary by your environment; guarded
    from langchain_openai import ChatOpenAI  # optional LangChain wrapper
except Exception:
    ChatOpenAI = None  # type: ignore

try:
    from langchain.embeddings import OpenAIEmbeddings  # optional
except Exception:
    OpenAIEmbeddings = None  # type: ignore

try:
    from langchain_community.vectorstores import Chroma  # optional
except Exception:
    Chroma = None  # type: ignore

# Optional direct OpenAI use
try:
    import openai
except Exception:
    openai = None  # type: ignore

# ----------------------------
# Configuration
# ----------------------------
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
DEFAULT_LLM = os.getenv("LLM_MODEL", "gpt-4o-mini")  # change as needed
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# When re-ranking via LLM, limit candidates to rerank_top_n
DEFAULT_RERANK_TOP_N = 10

# ----------------------------
# Helper: API key loader
# ----------------------------
def _get_openai_api_key() -> Optional[str]:
    # Streamlit secrets -> env
    if st and hasattr(st, "secrets"):
        k = st.secrets.get("OPENAI_API_KEY")
        if k:
            return k
    return os.getenv("OPENAI_API_KEY")

# ----------------------------
# Helper: cosine similarity
# ----------------------------
def cosine_sim_matrix(vecs: np.ndarray, qvec: np.ndarray) -> np.ndarray:
    """
    vecs: (N, D)
    qvec: (D,) or (1, D)
    returns similarity vector (N,)
    """
    if vecs is None or vecs.size == 0:
        return np.array([])
    q = qvec.reshape(-1)
    # normalize
    vecs_norm = np.linalg.norm(vecs, axis=1)
    q_norm = np.linalg.norm(q)
    denom = (vecs_norm * q_norm) + 1e-12
    sims = (vecs @ q) / denom
    return sims

# ----------------------------
# Fallback: embedding for query (if you need to compute it)
# This function tries to leverage the OpenAI embeddings (new or legacy).
# If not available, it raises.
# ----------------------------
def embed_query_openai(query: str, model: str = DEFAULT_EMBED_MODEL, api_key: Optional[str] = None) -> np.ndarray:
    api_key = api_key or _get_openai_api_key()
    if not api_key:
        raise RuntimeError("Missing OpenAI API key (for query embedding).")
    # prefer new openai client if installed
    try:
        # new openai package usage: openai.Embedding.create or OpenAI client
        if "OpenAI" in globals() and hasattr(globals().get("OpenAI"), "__call__"):
            # if new OpenAI client available in imports elsewhere
            pass
    except Exception:
        pass

    # Try legacy openai library if present
    try:
        if openai is None:
            raise RuntimeError("openai library not available")
        openai.api_key = api_key
        resp = openai.Embedding.create(model=model, input=[query])
        emb = np.array(resp["data"][0]["embedding"], dtype=np.float32)
        return emb
    except Exception as e:
        # try new OpenAI client if available
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.embeddings.create(model=model, input=[query])
            emb = np.array(resp.data[0].embedding, dtype=np.float32)
            return emb
        except Exception as e2:
            raise RuntimeError(f"Failed to compute embedding for query. openai errors: {e} / {e2}")

# ----------------------------
# LLM wrapper (safe initializers + call function)
# Will attempt to use LangChain ChatOpenAI if installed, otherwise openai.ChatCompletion
# ----------------------------
def init_llm_client(model_name: str = DEFAULT_LLM, temperature: float = LLM_TEMPERATURE):
    """Return a callable llm_invoke(messages: List[dict]) -> str"""
    api_key = _get_openai_api_key()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY for LLM")

    # If langchain ChatOpenAI available, use it (preferred for structured prompts)
    if ChatOpenAI is not None:
        try:
            # ChatOpenAI expects api_key param depending on wrapper
            llm = ChatOpenAI(api_key=api_key, model_name=model_name, temperature=temperature)
            def _invoke(messages: List[dict]) -> str:
                # expects {'role': 'user'/'system', 'content': '...'}
                try:
                    resp = llm(messages)  # LangChain ChatOpenAI may accept messages or use .generate; wrapper-dependent
                    # wrapper behaviour varies. Try to extract sensible text:
                    if hasattr(resp, "generations"):
                        # LangChain response object: extract text
                        gens = resp.generations
                        if gens and len(gens) > 0 and len(gens[0]) > 0:
                            return gens[0][0].text
                    # fallback string
                    return str(resp)
                except Exception as e:
                    raise RuntimeError(f"LangChain ChatOpenAI invocation failed: {e}")
            return _invoke
        except Exception:
            # fall through to direct openai
            pass

    # Fallback: direct openai.ChatCompletion or ChatCompletions (legacy/new)
    if openai is None:
        raise RuntimeError("Neither LangChain ChatOpenAI nor openai library are available for LLM calls.")

    # Set API key and create a simple invoker
    openai.api_key = api_key

    def _invoke(messages: List[dict]) -> str:
        """
        messages: list of dicts like [{"role": "system", "content": "..."}, {"role":"user","content":"..."}]
        returns assistant text or raises
        """
        try:
            # Try ChatCompletions endpoint
            resp = openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=1024,
            )
            # extract text
            if "choices" in resp and len(resp["choices"]) > 0:
                return resp["choices"][0]["message"]["content"].strip()
            return ""
        except Exception as e:
            # try completions as last resort
            try:
                resp = openai.Completion.create(
                    model=model_name,
                    prompt=messages[-1]["content"] if messages else "",
                    max_tokens=1024,
                    temperature=temperature,
                )
                if "choices" in resp and len(resp["choices"]) > 0:
                    return resp["choices"][0]["text"].strip()
                return ""
            except Exception as e2:
                raise RuntimeError(f"OpenAI invocation failed: {e} / {e2}")

    return _invoke

# ----------------------------
# Reranking: use LLM to score candidates
# We'll ask the LLM to return a JSON list of scores for each candidate (0-100).
# Provide robust parsing with fallback heuristics.
# ----------------------------
def llm_rerank_scores(query: str, candidates: Sequence[Tuple[str, Dict[str,Any]]], llm_invoke_callable, max_chars_per_candidate: int = 1200) -> List[float]:
    """
    candidates: sequence of (text, metadata) tuples
    returns list of scores (float) aligned with candidates order
    """
    # Build a succinct prompt: only include necessary snippet per candidate
    items_for_prompt = []
    for i, (text, meta) in enumerate(candidates):
        snippet = text.strip()
        if len(snippet) > max_chars_per_candidate:
            snippet = snippet[:max_chars_per_candidate] + "..."
        items_for_prompt.append({"idx": i, "text": snippet, "meta": meta})

    system = (
        "You are an assistant that scores relevance between a user question and short text chunks.\n"
        "Given a question and several candidate context chunks, return a JSON array of numeric relevance scores (0-100) "
        "in the same order as the chunks. Be concise and return ONLY valid JSON (e.g. [78, 12, 0, 55])."
    )

    # Build user content with the query and numbered candidate snippets
    user_lines = [f"Question: {query}", "", "Chunks:"]
    for it in items_for_prompt:
        user_lines.append(f"--- Chunk {it['idx']} ---")
        # include a short metadata hint if helpful
        if isinstance(it["meta"], dict) and "source" in it["meta"]:
            user_lines.append(f"(source: {it['meta'].get('source')})")
        user_lines.append(it["text"])
    user_content = "\n".join(user_lines)

    messages = [{"role":"system","content":system}, {"role":"user","content":user_content}]

    # Call LLM
    try:
        txt = llm_invoke_callable(messages)
        # try to find a JSON array in the response
        # naive parse: find first '[' and last ']' and json.loads
        start = txt.find('[')
        end = txt.rfind(']')
        if start != -1 and end != -1 and end > start:
            arr_text = txt[start:end+1]
            scores = json.loads(arr_text)
            # normalize to floats and length check
            if isinstance(scores, list) and len(scores) == len(candidates):
                return [float(s) for s in scores]
    except Exception as e:
        # swallow and use fallback
        pass

    # If LLM scoring failed, fallback to heuristic: similarity + token overlap
    fallback_scores = []
    for text, meta in candidates:
        # heuristic: score by number of shared words with query (case-insensitive) and length-adjusted
        q_words = set(w.lower() for w in query.split() if len(w) > 2)
        t_words = set(w.lower() for w in text.split() if len(w) > 2)
        common = q_words.intersection(t_words)
        sim_score = len(common) * 5.0  # modest weight
        # length penalty/bonus
        length_factor = min(30.0, len(text)/100.0)
        score = min(100.0, sim_score + length_factor)
        fallback_scores.append(score)
    return fallback_scores

# ----------------------------
# Retriever class
# ----------------------------
class Retriever:
    """
    Retriever that accepts in-memory arrays (texts, metadatas, vectors) or a Chroma in-memory store.
    Example:
        r = Retriever(texts=idx.texts, metadatas=idx.metadatas, vectors=idx.vectors)
        results = r.retrieve("what is policy for expense claims?", top_k=15, rerank=True)
    """

    def __init__(self,
                 texts: Optional[List[str]] = None,
                 metadatas: Optional[List[dict]] = None,
                 vectors: Optional[np.ndarray] = None,
                 chroma_store=None,
                 embed_model: str = DEFAULT_EMBED_MODEL,
                 llm_model: str = DEFAULT_LLM,
                 llm_temp: float = LLM_TEMPERATURE):
        """
        Provide either (texts + vectors) OR chroma_store. vectors must be numpy array shape (N, D).
        """
        self.texts = texts or []
        self.metadatas = metadatas or []
        self.vectors = vectors if vectors is None or isinstance(vectors, np.ndarray) else np.array(vectors)
        self.chroma = chroma_store
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.llm_temp = llm_temp

        # Pre-normalize vector norms for faster scoring if vectors provided
        self._norms = None
        if self.vectors is not None and self.vectors.size > 0:
            self._norms = np.linalg.norm(self.vectors, axis=1)

        # LLM invoker will be created on demand
        self._llm_invoke = None

    def _ensure_llm(self):
        if self._llm_invoke is None:
            try:
                self._llm_invoke = init_llm_client(model_name=self.llm_model, temperature=self.llm_temp)
            except Exception as e:
                # don't fail hard; record None
                self._llm_invoke = None
                _log = f"[WARN] LLM init failed: {e}"
                if st:
                    st.warning(_log)
                else:
                    print(_log)

    def retrieve(self, query: str, top_k: int = 15, rerank: bool = False, rerank_top_n: int = DEFAULT_RERANK_TOP_N) -> Dict[str, Any]:
        """
        Returns:
        {
            "query": query,
            "candidates": [ { "index": int, "score": float, "text": str, "metadata": dict }, ... ],
            "reranked": bool,
            "llm_answer": Optional[str]
        }
        """
        try:
            if not query or not query.strip():
                return {"error": "Empty query string."}

            # If chroma store is provided, use it first
            if self.chroma is not None:
                try:
                    docs = self.chroma.similarity_search(query, k=top_k)
                    candidates = []
                    for i, d in enumerate(docs):
                        txt = d.page_content if hasattr(d, "page_content") else getattr(d, "content", str(d))
                        meta = getattr(d, "metadata", {}) or {}
                        candidates.append({"index": i, "score": None, "text": txt, "metadata": meta})
                    return {"query": query, "candidates": candidates, "reranked": False, "llm_answer": None}
                except Exception as e:
                    print(f"[WARN] Chroma retrieval failed, falling back to numpy: {e}")

            # Validate in-memory vectors/texts
            if self.vectors is None or self.vectors.size == 0 or not self.texts:
                return {"error": "No in-memory vectors/texts available for retrieval."}

            # Determine vector dimensionality
            try:
                vec_dim = int(self.vectors.shape[1])
            except Exception:
                return {"error": "Invalid vector matrix shape; cannot determine embedding dimensionality."}

            # Map embedding dimension -> canonical model name(s)
            # If you used a different model, pass embed_model when instantiating Retriever
            model_candidates_by_dim = {
                3072: ["text-embedding-3-large", "text-embedding-3-small"],
                1536: ["text-embedding-3-small", "text-embedding-3-large"],
            }

            # Decide initial model to request for the query embedding
            model_to_try = None
            if self.embed_model and isinstance(self.embed_model, str) and self.embed_model.strip():
                model_to_try = self.embed_model
            else:
                model_to_try = model_candidates_by_dim.get(vec_dim, [self.embed_model])[0] if vec_dim in model_candidates_by_dim else self.embed_model

            # Try to compute query embedding; if dimension mismatch, try fallback models
            qvec = None
            last_err = None

            # Build candidate list to try: prefer model_to_try first, then dimension-informed candidates
            try_models = []
            if model_to_try:
                try_models.append(model_to_try)
            if vec_dim in model_candidates_by_dim:
                for m in model_candidates_by_dim[vec_dim]:
                    if m not in try_models:
                        try_models.append(m)
            # ensure we have at least a couple of sensible fallbacks
            for fallback in ["text-embedding-3-large", "text-embedding-3-small"]:
                if fallback not in try_models:
                    try_models.append(fallback)

            for m in try_models:
                try:
                    qvec_raw = embed_query_openai(query, model=m, api_key=_get_openai_api_key())
                    if not isinstance(qvec_raw, np.ndarray):
                        qvec = np.array(qvec_raw, dtype=np.float32)
                    else:
                        qvec = qvec_raw.astype(np.float32)
                    # If qvec is 2D like (1,D), convert to 1D
                    if qvec.ndim == 2 and qvec.shape[0] == 1:
                        qvec = qvec.reshape(-1)
                    # Check dimension
                    if qvec.shape[0] == vec_dim:
                        # success
                        used_model = m
                        break
                    else:
                        last_err = f"Embedding model '{m}' produced dimension {qvec.shape[0]} (expected {vec_dim})"
                        qvec = None
                except Exception as e:
                    last_err = f"Embedding call failed for model '{m}': {e}"
                    qvec = None
                    # try next model

            if qvec is None:
                # Could not generate a matching-dimension embedding. Provide a helpful error message.
                msg = (
                    "Failed to compute a query embedding that matches the index embedding dimension.\n"
                    f"Index embedding dim = {vec_dim}. Tried models: {try_models}.\n"
                    f"Last error: {last_err}\n\n"
                    "Actions:\n"
                    "- Rebuild the index using the same embedding model you will use at query time (e.g. text-embedding-3-large) OR\n"
                    "- Instantiate Retriever with embed_model that matches your index (Retriever(..., embed_model='text-embedding-3-large'))\n"
                )
                return {"error": msg}

            # 2) compute cosine similarities
            sims = cosine_sim_matrix(self.vectors, qvec)
            if sims.size == 0:
                return {"error": "Embeddings present but similarity computation failed."}

            # 3) pick top_k indices
            top_k = min(int(top_k), len(sims))
            top_idx = np.argsort(-sims)[:top_k]
            candidates = []
            for idx in top_idx:
                txt = self.texts[int(idx)]
                meta = self.metadatas[int(idx)] if len(self.metadatas) > int(idx) else {}
                candidates.append({"index": int(idx), "score": float(sims[int(idx)]), "text": txt, "metadata": meta})

            # 4) optional re-ranking
            reranked_flag = False
            if rerank and len(candidates) > 0:
                # prepare (text, meta) list for reranker
                candidate_pairs = [(c["text"], c["metadata"]) for c in candidates[:rerank_top_n]]
                self._ensure_llm()
                if self._llm_invoke is not None:
                    try:
                        scores = llm_rerank_scores(query, candidate_pairs, self._llm_invoke)
                        # attach scores and sort by them (desc)
                        for i, s in enumerate(scores):
                            candidates[i]["score_rerank"] = float(s)
                        candidates = sorted(candidates, key=lambda x: x.get("score_rerank", x["score"]), reverse=True)
                        reranked_flag = True
                    except Exception as e:
                        print(f"[WARN] LLM rerank failed: {e}\n{traceback.format_exc()}")
                else:
                    print("[WARN] LLM not available for reranking; skipping rerank.")

            return {
                "query": query,
                "candidates": candidates,
                "reranked": reranked_flag,
                "llm_answer": None
            }

        except Exception as e:
            return {"error": f"Retriever failure: {e}\n{traceback.format_exc()}"}


# ----------------------------
# Convenience function: retrieval + optional QA with LLM on combined context
# returns a tuple (answer_text, candidates_texts)
# ----------------------------
def policy_handler_from_retriever(
    retriever: Retriever,
    query: str,
    top_k: int = 15,
    rerank: bool = True,
    llm_model: Optional[str] = None,
    llm_temperature: float = 0.0
) -> Tuple[str, List[str]]:
    """
    Retrieves top_k candidates (with optional rerank), then calls LLM to produce
    a concise answer from the retrieved context.
    Returns (answer_text, context_chunk_list).
    """

    # 1) Retrieve from vector index
    res = retriever.retrieve(
        query=query,
        top_k=top_k,
        rerank=rerank,
        rerank_top_n=min(top_k, DEFAULT_RERANK_TOP_N)
    )

    if "error" in res:
        return (f"ERROR: {res['error']}", [])

    candidates = res.get("candidates", [])
    if not candidates:
        return ("No relevant context found.", [])

    # 2) Build combined context
    context_texts = []
    for c in candidates:
        meta = c.get("metadata", {}) or {}
        source = meta.get("source") or meta.get("file") or meta.get("collection") or ""
        prefix = f"[source: {source}] " if source else ""
        context_texts.append(prefix + (c.get("text") or ""))

    combined_context = "\n\n---\n\n".join(context_texts)

    # 3) Initialize LLM
    try:
        llm_invoke = init_llm_client(
            model_name=llm_model or retriever.llm_model,
            temperature=llm_temperature
        )
    except Exception as e:
        return (f"LLM unavailable: {e}", context_texts)

    # 4) Prepare the QA prompt
    system_prompt = (
        "You must answer strictly from the provided context. "
        "If the answer is not present in the context, reply with: "
        "'I don't have enough information to answer that question.'"
    )

    user_prompt = (
        f"Context:\n{combined_context}\n\n"
        f"Question:\n{query}\n\n"
        "Give the most accurate and concise answer using ONLY the context above."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # 5) Generate answer
    try:
        answer = llm_invoke(messages)
        return (answer.strip(), context_texts)
    except Exception as e:
        return (f"LLM call failed: {e}", context_texts)



# ----------------------------
# Small test harness if run as script
# ----------------------------
if __name__ == "__main__":
    print("Retriever module test (quick sanity)")

    # Quick synthetic test if environment has embeddings, otherwise skip
    # Example: try to load an index file if exists (expected: you pass arrays from RAGIndexer)
    try:
        # look for saved sample npy / json in current dir (optional)
        if Path("saved_index_vectors.npy").exists() and Path("saved_index_texts.json").exists():
            vecs = np.load("saved_index_vectors.npy")
            texts = json.loads(Path("saved_index_texts.json").read_text(encoding="utf-8"))
            metas = []
            if Path("saved_index_metas.json").exists():
                metas = json.loads(Path("saved_index_metas.json").read_text(encoding="utf-8"))
            r = Retriever(texts=texts, metadatas=metas, vectors=vecs)
            q = input("Enter test query: ").strip()
            out = r.retrieve(q, top_k=15, rerank=False)
            print(json.dumps(out, indent=2, ensure_ascii=False))
        else:
            print("No saved_index_*.npy/json found; instantiate Retriever in code with your RAGIndexer outputs.")
    except Exception as e:
        print("Retriever self-test failed:", e)
