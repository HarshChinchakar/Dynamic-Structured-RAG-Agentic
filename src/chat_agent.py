# src/chat_agent.py

import os
import traceback
from typing import List, Dict, Tuple

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# --- Robust imports (match main.py behavior) ---
# retriever: try local module name (because main.py's loader injects both "src.retrival_class" and "retrival_class")
try:
    from retrival_class import Retriever
except Exception:
    try:
        # Fallback if someone imports this file directly without main.py's loader
        from src.retrival_class import Retriever  # type: ignore
    except Exception as e:
        # Make the reason visible in Streamlit and during import
        raise ImportError(
            "chat_agent.py: Failed to import Retriever. "
            "Ensure 'src/retrival_class.py' exists and defines class 'Retriever'. "
            f"Underlying error: {e}"
        )

# langchain memory (present in your requirements)
try:
    from langchain.memory import ChatMessageHistory
except Exception as e:
    raise ImportError(
        "chat_agent.py: Failed to import ChatMessageHistory from langchain. "
        "Check your langchain installation/version."
    ) from e

# openai client (your requirements use openai>=2.x which supports Responses API)
try:
    from openai import OpenAI
except Exception as e:
    raise ImportError(
        "chat_agent.py: Failed to import OpenAI client. Check 'openai' package."
    ) from e


# ============================
# Config (aligned with multimedia.py)
# ============================
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.2
MAX_TOKENS = 1024
CHUNK_CHAR_LIMIT = 1200
HISTORY_LIMIT = 3   # last 3 user+assistant turns (6 messages total)


def _extract_text(resp):
    """Exactly the same extraction pattern you use elsewhere."""
    try:
        c = resp.output_text
        if isinstance(c, str):
            return c.strip()
        return str(c)
    except Exception:
        return str(resp)


def _get_api_key() -> str:
    """Match multimedia.py behavior; show a clear error if missing."""
    api_key = None
    try:
        if st and hasattr(st, "secrets"):
            api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in Streamlit secrets or environment."
        )

    return api_key


class ChatAgent:
    """
    Policy RAG chat with short-term memory (last 3 Q/A pairs).
    Expects 'rag_cache' from Tab 1: {'texts','vectors','metadatas','embed_model'}
    """

    def __init__(self, rag_cache: dict):
        # Validate rag_cache early with explicit errors
        missing = [k for k in ("texts", "vectors", "metadatas", "embed_model") if k not in rag_cache]
        if missing:
            raise ValueError(f"chat_agent.ChatAgent: rag_cache missing keys: {missing}")

        self.rag_cache = rag_cache

        # Retriever
        try:
            self.retriever = Retriever(
                texts=rag_cache["texts"],
                vectors=rag_cache["vectors"],
                metadatas=rag_cache["metadatas"],
                embed_model=rag_cache["embed_model"],
            )
        except Exception as e:
            raise RuntimeError(f"chat_agent.ChatAgent: Failed to create Retriever: {e}")

        # In-RAM chat history
        try:
            self.history = ChatMessageHistory()
        except Exception as e:
            raise RuntimeError(f"chat_agent.ChatAgent: Failed to init ChatMessageHistory: {e}")

        # OpenAI client
        try:
            self.client = OpenAI(api_key=_get_api_key())
        except Exception as e:
            raise RuntimeError(f"chat_agent.ChatAgent: Failed to init OpenAI: {e}")

    # ------------------ history helpers ------------------
    def _add_to_history(self, role: str, content: str):
        if role == "user":
            self.history.add_user_message(content)
        else:
            self.history.add_ai_message(content)

    def _recent_pairs_as_bullets(self) -> List[str]:
        # Last 3 user+assistant pairs (6 messages)
        msgs = self.history.messages[-HISTORY_LIMIT * 2:]
        bullets = []
        for m in msgs:
            role = "USER" if getattr(m, "type", "human") == "human" else "ASSISTANT"
            text = getattr(m, "content", "")
            bullets.append(f"{role}: {text}")
        return bullets

    # ------------------ retrieval ------------------
    def _retrieve_chunks(self, query: str, top_k: int = 8):
        try:
            ret = self.retriever.retrieve(query, top_k=top_k, rerank=True)
        except Exception as e:
            return [], {"error": f"Retriever crashed: {e}"}

        if isinstance(ret, dict) and "error" in ret:
            return [], ret

        candidates = ret.get("candidates", []) if isinstance(ret, dict) else []
        chunks = []
        for c in candidates:
            t = (c.get("text") or "").strip()
            if t:
                if len(t) > CHUNK_CHAR_LIMIT:
                    t = t[:CHUNK_CHAR_LIMIT] + "...[truncated]"
                chunks.append(t)

        return chunks, ret

    # ------------------ prompt ------------------
    def _build_prompt(self, query: str, history_bullets: List[str], chunks: List[str]) -> str:
        history_text = "\n".join(history_bullets) if history_bullets else "No prior conversation."
        context = "\n---\n".join(chunks)

        return f"""
You are an HR Policy Assistant.
Answer questions STRICTLY using the provided policy context and the allowed inferences.

You ARE allowed to make logical inferences if clearly implied.
You are NOT allowed to hallucinate missing details.
If the answer cannot be found or inferred, reply EXACTLY:
"I don't have enough information in the provided documents."

-----------------------------------
RECENT CHAT MEMORY:
{history_text}
-----------------------------------
POLICY CONTEXT:
{context}
-----------------------------------
USER QUESTION:
{query}
-----------------------------------

Now give the best possible answer based only on the context and allowed inferences.
""".strip()

    # ------------------ public API ------------------
    def chat_turn(self, user_query: str) -> Tuple[str, List[Dict], Dict]:
        # 1) history update
        self._add_to_history("user", user_query)

        # 2) memory
        mem_bullets = self._recent_pairs_as_bullets()

        # 3) retrieve
        chunks, retrieval_meta = self._retrieve_chunks(user_query)
        if not chunks:
            reply = "I don't have enough information in the provided documents."
            self._add_to_history("assistant", reply)
            return reply, self._history_as_list(), {"retrieval": retrieval_meta, "used_chunks": [], "used_history": mem_bullets}

        # 4) prompt -> LLM
        prompt = self._build_prompt(user_query, mem_bullets, chunks)
        try:
            resp = self.client.responses.create(
                model=MODEL_NAME,
                input=prompt,
                max_output_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            reply = _extract_text(resp)
        except Exception as e:
            reply = f"[ERROR] LLM failed: {e}\n{traceback.format_exc()}"

        # 5) save assistant turn
        self._add_to_history("assistant", reply)

        return reply, self._history_as_list(), {
            "retrieval": retrieval_meta,
            "used_chunks": chunks,
            "used_history": mem_bullets,
        }

    def _history_as_list(self) -> List[Dict]:
        out = []
        for m in self.history.messages:
            role = "user" if getattr(m, "type", "human") == "human" else "assistant"
            out.append({"role": role, "content": getattr(m, "content", "")})
        return out

    def clear(self):
        try:
            self.history = ChatMessageHistory()
        except Exception:
            self.history = ChatMessageHistory()


__all__ = ["ChatAgent"]
