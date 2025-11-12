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
TEMPERATURE = 0.0
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

        # ORIGINAL QUERY FOR REFERENCE
        # You are an HR Policy Assistant.  
            # Your job is to answer using ONLY the information found inside the provided context.
            
            #  You ARE allowed to make **logical inferences** when the information is implied:
            #    - Example: If policy states "All full-time employees receive X", you may infer
            #      the benefit applies to any full-time employee even if not stated explicitly.
            #    - Example: If a rule describes reimbursement rules, you may apply the rule to
            #      similar expense cases even if that specific example is not shown.
            
            # [IMPORTANT] You are NOT allowed to hallucinate missing details.
            # [IMPORTANT] Never invent numbers, dates, names, amounts, or policy clauses that are not present or inferable.
            # [IMPORTANT] If the answer cannot be found or inferred from the context, reply EXACTLY:
            
            # "I don't have enough information in the provided documents."
            
            # Do NOT reveal any confidential, sensitive, or private information.
            # Keep answers strictly within HR & policy interpretation boundaries.
            # nsure responses are clear, accurate, and concise.
            
            # -----------------------------------
            # CONTEXT:
            # {context}
            # -----------------------------------
            
            # QUESTION:
            # {query}
            
            # Now produce the **best possible answer using the context and valid inferences only**.
            # If insufficient context is available, reply with the exact fallback sentence.
            

        # New hallucination-safe, inference-friendly prompt
        prompt = f"""
            You are an HR Policy Assistant. Your job is to answer the user's QUESTION using ONLY the information present in the provided CONTEXT or by making narrow, logical inferences that are directly supported by that context.

            RESPONSE RULES (follow exactly)
            1. INTERNAL EVIDENCE CHECK (do this mentally — do NOT output it):  
               - First, scan the entire CONTEXT and locate any sentence(s) that directly answer the QUESTION or together imply an answer. You may combine multiple sentences only when they together make the required inference logically entailed by the text. Do NOT invent or assume facts that are not present or logically entailed.  
               - Do NOT output any intermediate reasoning, search steps, chunk ids, filenames, or quotes from the context. Internal evidence must remain hidden.
            
            2. WHEN TO ANSWER vs FALLBACK:  
               - If the answer is explicitly present in the CONTEXT, or can be **clearly** and **conservatively** inferred from the CONTEXT, produce a concise user-facing answer (see Output Style below).  
               - If the CONTEXT does not contain enough information to support an answer or a conservative inference, reply EXACTLY and ONLY with the following fallback sentence (no extra text, no punctuation changes):  
                 "I don't have enough information in the provided documents."
            
            3. INFERENCE GUIDELINES (allowed, but conservative):  
               - Allowed: logical generalization (e.g., if CONTEXT says "All full-time employees receive X", you may state X applies to full-time employees).  
               - Allowed: combining multiple explicit statements to derive a clear conclusion when all steps are entailed by the text.  
               - Not allowed: inventing specific numbers, dates, names, monetary amounts, thresholds, or procedural steps unless they appear in the CONTEXT or are the only conservative inference. If a numeric value is not explicitly present and cannot be conservatively inferred, use the fallback.  
               - When you perform an allowed inference, ensure it is minimal and directly supported (do not overextend).
            
            4. SPECIAL NUANCES:
               - For questions asking for lists of policy types (e.g., "leave types"), by default return the most common / primary items (up to 5) unless the user explicitly asked for a complete or exhaustive list. If the CONTEXT only lists all items explicitly, you may list them; otherwise pick the most common ones implied by the CONTEXT.
               - Do not ask clarifying questions. Provide the best possible answer permitted by the CONTEXT and these rules.
            
            5. OUTPUT STYLE (required):
               - Produce ONLY the user-facing answer text. Do NOT include any citations, chunk ids, filenames, debugging info, internal notes, or evidence excerpts.  
               - Keep answers concise and clear. Prefer 1–3 short paragraphs or a short bullet list (no more than ~7 bullets) when relevant.  
               - If returning a simple factual value (e.g., "3 or more"), answer directly (e.g., `3 or more`). If additional clarity helps, add one short clarifying sentence, but no provenance.
            
            6. TONE:
               - Professional, neutral, concise, and policy-focused.
            
            -----------------------------------
            CONTEXT:
            {context}
            -----------------------------------
            QUESTION:
            {query}
            
            Now produce the answer following the RESPONSE RULES and OUTPUT STYLE above.

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
