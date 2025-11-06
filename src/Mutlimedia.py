#!/usr/bin/env python3
"""
multimedia.py  (FAST VERSION)

Ultra-optimized final answer generator.
Uses the new OpenAI Responses API (much faster and very stable).
Shorter prompt, smaller token usage, trimmed chunks.
"""

import os
import traceback
from dotenv import load_dotenv

# Optional Streamlit import
try:
    import streamlit as st
except Exception:
    st = None

from openai import OpenAI

# ============================
# 🔧 CONFIG
# ============================
load_dotenv()

MODEL_NAME = "gpt-4o-mini"        # Fast + strong
TEMPERATURE = 0.2
MAX_TOKENS = 1024                 # Smaller = faster

# Limit context size per chunk
CHUNK_CHAR_LIMIT = 1200           # Reduce to 800 for even more speed


# ============================
# ✅ API KEY HANDLING
# ============================
def get_api_key():
    api_key = None

    if st and hasattr(st, "secrets"):
        api_key = st.secrets.get("OPENAI_API_KEY")

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    return api_key


# ============================
# ✅ SAFE EXTRACTOR
# ============================
def extract_text(resp):
    """
    Safely extract content from Responses API output.
    """
    try:
        c = resp.output_text
        if isinstance(c, str):
            return c.strip()
        return str(c)
    except:
        return str(resp)


# ============================
# ✅ FAST MULTIMEDIA RESPONSE
# ============================
# def multimedia_response(query: str, context_chunks: list[str]) -> str:
#     """
#     FAST version:
#     - Uses OpenAI Responses API
#     - Shorter context
#     - Trims each chunk
#     - Much lower latency
#     """

#     try:
#         api_key = get_api_key()
#         client = OpenAI(api_key=api_key)

#         # Trim long chunks
#         trimmed_chunks = []
#         for c in context_chunks:
#             c = c.strip()
#             if len(c) > CHUNK_CHAR_LIMIT:
#                 c = c[:CHUNK_CHAR_LIMIT] + "...[truncated]"
#             trimmed_chunks.append(c)

#         context = "\n---\n".join(trimmed_chunks)

#         prompt = f"""
# Answer the question ONLY using the context below.

# If the answer is not explicitly present, reply exactly:
# "I don't have enough information in the provided documents."

# CONTEXT:
# {context}

# QUESTION: {query}

# Answer concisely.
# """

#         # ✅ This API is the fastest available
#         response = client.responses.create(
#             model=MODEL_NAME,
#             input=prompt,
#             max_output_tokens=MAX_TOKENS,
#             temperature=TEMPERATURE
#         )

#         return extract_text(response)

#     except Exception as e:
#         return f"[ERROR] multimedia_response failed: {e}\n{traceback.format_exc()}"

def multimedia_response(query: str, context_chunks: list[str]) -> str: 
    """
    Policy-aware, hallucination-safe multimedia response generator.
    Allows INDIRECT inference from policy context.
    Strictly prevents hallucinations.
    """

    try:
        api_key = get_api_key()
        client = OpenAI(api_key=api_key)

        # Trim long chunks
        trimmed_chunks = []
        for c in context_chunks:
            c = c.strip()
            if len(c) > CHUNK_CHAR_LIMIT:
                c = c[:CHUNK_CHAR_LIMIT] + "...[truncated]"
            trimmed_chunks.append(c)

        context = "\n---\n".join(trimmed_chunks)

        # New hallucination-safe, inference-friendly prompt
        prompt = f"""
            You are an HR Policy Assistant.  
            Your job is to answer using ONLY the information found inside the provided context.
            
             You ARE allowed to make **logical inferences** when the information is implied:
               - Example: If policy states "All full-time employees receive X", you may infer
                 the benefit applies to any full-time employee even if not stated explicitly.
               - Example: If a rule describes reimbursement rules, you may apply the rule to
                 similar expense cases even if that specific example is not shown.
            
            [IMPORTANT] You are NOT allowed to hallucinate missing details.
            [IMPORTANT] Never invent numbers, dates, names, amounts, or policy clauses that are not present or inferable.
            [IMPORTANT] If the answer cannot be found or inferred from the context, reply EXACTLY:
            
            "I don't have enough information in the provided documents."
            
            Do NOT reveal any confidential, sensitive, or private information.
            Keep answers strictly within HR & policy interpretation boundaries.
            nsure responses are clear, accurate, and concise.
            
            -----------------------------------
            CONTEXT:
            {context}
            -----------------------------------
            
            QUESTION:
            {query}
            
            Now produce the **best possible answer using the context and valid inferences only**.
            If insufficient context is available, reply with the exact fallback sentence.
            """

        #  Call the fast Responses API
        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt,
            max_output_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )

        return extract_text(response)

    except Exception as e:
        return f"[ERROR] multimedia_response failed: {e}\n{traceback.format_exc()}"


# ============================
# ✅ STANDALONE TESTING
# ============================
if __name__ == "__main__":
    print("\n=== FAST MULTIMEDIA TEST ===\n")

    q = input("Query:\n> ").strip()

    print("\nEnter context chunks (finish with empty line):")
    chunks = []
    while True:
        line = input()
        if not line.strip():
            break
        chunks.append(line)

    print("\n[RUNNING FAST LLM]\n")
    ans = multimedia_response(q, chunks)

    print("\n========= ANSWER =========\n")
    print(ans)
    print("\n==========================\n")
