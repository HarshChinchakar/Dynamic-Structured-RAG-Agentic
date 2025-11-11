# src/chat_agent.py

import os
import traceback
from datetime import datetime
from typing import List, Dict, Tuple
import streamlit as st


import streamlit as st                         # ✅ REQUIRED (you used st in get_api_key)
from openai import OpenAI
from langchain.memory import ChatMessageHistory

# ✅ Import retriever EXACTLY like main.py loads it (local import)
from retrival_class import Retriever

from dotenv import load_dotenv
load_dotenv()

def get_api_key():
    api_key = None

    if st and hasattr(st, "secrets"):
        api_key = st.secrets.get("OPENAI_API_KEY")

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    return api_key


# ============================================================
# ✅ FAST LLM CONFIG  (same extraction logic as multimedia.py)
# ============================================================
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.2
MAX_TOKENS = 1024
CHUNK_CHAR_LIMIT = 1200
HISTORY_LIMIT = 3   # ✅ last 3 turns (user+assistant) => 6 messages max


def extract_text(resp):
    """Exactly same extraction structure as multimedia.py"""
    try:
        c = resp.output_text
        if isinstance(c, str):
            return c.strip()
        return str(c)
    except:
        return str(resp)


# ============================================================
# ✅ ChatAgent (Policy RAG + Chat Memory)
# ============================================================
class ChatAgent:

    def __init__(self, rag_cache: dict):
        """
        rag_cache → taken from st.session_state.rag_cache
        """
        self.rag_cache = rag_cache
        self.retriever = Retriever(
            texts=rag_cache["texts"],
            vectors=rag_cache["vectors"],
            metadatas=rag_cache["metadatas"],
            embed_model=rag_cache["embed_model"],
        )

        # ✅ in-RAM chat history (LangChain)
        self.history = ChatMessageHistory()

        # ✅ OpenAI client
        self.api_key = get_api_key()
        self.client = OpenAI(api_key=self.api_key)

    # ---------------------------------------------------------
    # ✅ Add message to LC history
    # ---------------------------------------------------------
    def _add_to_history(self, role: str, content: str):
        if role == "user":
            self.history.add_user_message(content)
        else:
            self.history.add_ai_message(content)

    # ---------------------------------------------------------
    # ✅ Get last N chat turns (user+assistant)
    # ---------------------------------------------------------
    def _get_recent_chat_pairs(self) -> List[str]:
        all_msgs = self.history.messages

        # keep last 6 messages (3 user + 3 assistant)
        recent = all_msgs[-HISTORY_LIMIT * 2:]

        formatted = []
        for m in recent:
            role = "USER" if m.type == "human" else "ASSISTANT"
            formatted.append(f"{role}: {m.content}")

        return formatted

    # ---------------------------------------------------------
    # ✅ Retrieve relevant policy chunks
    # ---------------------------------------------------------
    def _retrieve_chunks(self, query: str, top_k: int = 8):
        try:
            ret = self.retriever.retrieve(query, top_k=top_k, rerank=True)
        except Exception as e:
            return [], {"error": str(e)}

        if "error" in ret:
            return [], ret

        candidates = ret.get("candidates", [])
        chunks = []

        for c in candidates:
            t = c.get("text", "").strip()
            if len(t) > CHUNK_CHAR_LIMIT:
                t = t[:CHUNK_CHAR_LIMIT] + "...[truncated]"
            chunks.append(t)

        return chunks, ret

    # ---------------------------------------------------------
    # ✅ Build new LLM prompt (updated multimedia-style prompt)
    # ---------------------------------------------------------
    def _build_prompt(self, query: str, recent_history: List[str], chunks: List[str]) -> str:

        history_text = "\n".join(recent_history) if recent_history else "No prior conversation."

        context = "\n---\n".join(chunks)

        prompt = f"""
You are an HR Policy Assistant.  
Answer questions **STRICTLY using the provided policy context** and the allowed inferences.

You ARE allowed to make **logical inferences** if clearly implied.
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
"""

        return prompt

    # ---------------------------------------------------------
    # ✅ Main function used by Tab 4
    # ---------------------------------------------------------
    def chat_turn(self, user_query: str) -> Tuple[str, List[Dict], Dict]:
        """
        user_query → the message from UI
        Returns: reply_text, updated_history, debug_info
        """

        # ✅ Step 1: add user query to history
        self._add_to_history("user", user_query)

        # ✅ Step 2: get last 3 query+response pairs
        recent_history = self._get_recent_chat_pairs()

        # ✅ Step 3: retrieve policy text chunks
        chunks, retrieval_meta = self._retrieve_chunks(user_query)

        if not chunks:
            reply = "I don't have enough information in the provided documents."
            self._add_to_history("assistant", reply)
            return reply, self._history_as_list(), {"retrieval": retrieval_meta}

        # ✅ Step 4: build the LLM prompt
        prompt = self._build_prompt(user_query, recent_history, chunks)

        # ✅ Step 5: LLM call -- same extraction style as multimedia.py
        try:
            response = self.client.responses.create(
                model=MODEL_NAME,
                input=prompt,
                max_output_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            reply = extract_text(response)

        except Exception as e:
            reply = f"[ERROR] LLM failed: {e}\n{traceback.format_exc()}"

        # ✅ Step 6: add assistant reply to history
        self._add_to_history("assistant", reply)

        return reply, self._history_as_list(), {
            "retrieval": retrieval_meta,
            "used_chunks": chunks,
            "used_history": recent_history,
        }

    # ---------------------------------------------------------
    # ✅ Convert LC chat history into list for main.py
    # ---------------------------------------------------------
    def _history_as_list(self):
        result = []
        for m in self.history.messages:
            role = "user" if m.type == "human" else "assistant"
            result.append({"role": role, "content": m.content})
        return result

    # ---------------------------------------------------------
    # ✅ Wipe session (optional)
    # ---------------------------------------------------------
    def clear(self):
        self.history = ChatMessageHistory()
