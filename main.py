

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ DIFFERENT TABS

import streamlit as st
from datetime import datetime, timezone
import traceback
import os
import sys
import importlib
import importlib.util

# ------------------------------------------------
# PATH SETUP
# ------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")


# ------------------------------------------------
# SILENT MODULE LOADER (NO PRINTS)
# ------------------------------------------------
def load_src_module(module_name: str):

    full_name = f"src.{module_name}"

    # 1. Try normal import
    try:
        return importlib.import_module(full_name)
    except Exception:
        pass

    # 2. Fallback: load raw file
    module_path = os.path.join(SRC_DIR, f"{module_name}.py")

    if not os.path.isfile(module_path):
        raise ImportError(f"Module file not found: {module_path}")

    spec = importlib.util.spec_from_file_location(full_name, module_path)
    mod = importlib.util.module_from_spec(spec)

    sys.modules[full_name] = mod
    sys.modules[module_name] = mod

    try:
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        raise


# ------------------------------------------------
# IMPORTS FOR TAB 1 (Policy RAG)
# ------------------------------------------------
try:
    Router_mod = load_src_module("Router_gpt")
    classify_query = getattr(Router_mod, "classify_query")
except Exception:
    classify_query = None

try:
    Emb_mod = load_src_module("embedding_Class")
    RAGIndexer = getattr(Emb_mod, "RAGIndexer")
except Exception:
    st.error("Failed loading embedding_Class:")
    st.stop()

try:
    Ret_mod = load_src_module("retrival_class")
    Retriever = getattr(Ret_mod, "Retriever")
    policy_handler_from_retriever = getattr(Ret_mod, "policy_handler_from_retriever", None)
except Exception:
    st.error("Failed loading retrival_class:")
    st.stop()

try:
    Multi_mod = load_src_module("Mutlimedia")
    multimedia_response = getattr(Multi_mod, "multimedia_response", None)
except Exception:
    multimedia_response = None


# ------------------------------------------------
# IMPORT app.py (Mongo Agent)
# ------------------------------------------------
run_document_query = None
try:
    App_mod = load_src_module("app")
    if hasattr(App_mod, "run_document_query"):
        run_document_query = getattr(App_mod, "run_document_query")
except Exception:
    pass


# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="Policy + Mongo Agent", page_icon="🧠", layout="wide")
st.title("🧠 AI Assistant — Policy RAG + Mongo Document Agent")


# ------------------------------------------------
# TABS
# ------------------------------------------------
tab1, tab2 = st.tabs(["📘 Policy RAG (Existing Debug Mode)", "🗄️ Document Query — Mongo Agent"])


# ------------------------------------------------
# ✅ TAB 1 — UNCHANGED
# ------------------------------------------------
with tab1:
    st.header("📘 Policy RAG — DEBUG MODE (Unchanged)")

    # STATE INIT
    if "rag_cache" not in st.session_state:
        st.session_state.rag_cache = None

    if "query_to_run" not in st.session_state:
        st.session_state.query_to_run = None

    # INPUTS
    user_query = st.text_area("Enter your question", height=150)
    run = st.button("Run Query (Policy Only)")

    rebuild = st.button("Rebuild Embeddings (force)")
    if rebuild:
        st.session_state.rag_cache = None
        st.info("Cache cleared, embeddings will rebuild on next Run.")

    POLICIES_PATH = os.path.join(ROOT_DIR, "Dataset", "Policies")

    # FUNCTION
    def build_index_debug():
        try:
            idx = RAGIndexer(
                local_paths=[POLICIES_PATH],
                s3_urls=None,
                workdir="rag_work",
                embed_model="text-embedding-3-large",
                max_tokens=900,
                overlap=150,
                min_chunk_chars=280,
            )

            idx.build()

            st.session_state.rag_cache = {
                "texts": idx.texts,
                "vectors": idx.vectors,
                "metadatas": idx.metadatas,
                "embed_model": idx.cfg.embed_model,
            }

            st.success("Embedding SUCCESS — stored to RAM")

        except Exception:
            st.error("Embedding failed:")
            st.code(traceback.format_exc())

    if st.session_state.rag_cache is None:
        build_index_debug()

    if run:
        if not user_query.strip():
            st.warning("Enter a valid query.")
            st.stop()
        st.session_state.query_to_run = user_query.strip()

    if st.session_state.query_to_run:
        q = st.session_state.query_to_run

        st.markdown("---")
        st.header("DEBUG EXECUTION — POLICY ONLY")

        cache = st.session_state.rag_cache

        try:
            retr = Retriever(
                texts=cache["texts"],
                vectors=cache["vectors"],
                metadatas=cache["metadatas"],
                embed_model=cache["embed_model"],
            )
        except Exception:
            st.error("Retriever creation failed:")
            st.code(traceback.format_exc())
            st.stop()

        try:
            ret = retr.retrieve(q, top_k=10, rerank=True)
        except Exception:
            st.error("Retriever failed:")
            st.code(traceback.format_exc())
            st.stop()

        st.json(ret)

        if "error" in ret:
            st.error("Retriever returned error:", ret["error"])
            st.stop()

        candidates = ret.get("candidates", [])
        chunks = [c["text"] for c in candidates]

        with st.expander("📄 Retrieved Chunks (Click to Expand)", expanded=False):
        for i, c in enumerate(chunks):
            st.markdown(f"### Chunk {i+1}")
            st.code(c)


        st.header("LLM ANSWER — DEBUG MODE")

        try:
            if multimedia_response:
                final_ans = multimedia_response(q, chunks)
            else:
                final_ans = "\n\n-----------\n\n".join(chunks)
        except Exception:
            st.error("LLM Answer generation failed:")
            st.code(traceback.format_exc())
            final_ans = f"[ERROR] {e}"

        st.subheader("FINAL ANSWER")
        st.write(final_ans)

        st.session_state.query_to_run = None


# ------------------------------------------------
# ✅ TAB 2 — Mongo Agent (Clean Mode)
# ------------------------------------------------
with tab2:

    st.header("🗄️ Mongo HR Agent — Clean Mode")

    email = st.text_input("User Email")
    query = st.text_area("Document Query", height=150)

    if st.button("Run Document Query"):

        if not email.strip() or not query.strip():
            st.warning("Please enter BOTH Email and Query.")
            st.stop()

        import subprocess, shlex

        uv_cmd = [
            "uv", "run",
            "src/app.py",
            "--email", email,
            "--query", query
        ]

        logs = []

        try:
            proc = subprocess.Popen(
                uv_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=ROOT_DIR,
                bufsize=1
            )
        except Exception as e:
            st.error("Failed to run uv.")
            st.code(str(e))
            st.stop()

        for line in proc.stdout:
            logs.append(line.rstrip("\n"))

        proc.wait()

        st.subheader("Execution Log")
        st.code("\n".join(logs))

        # Extract final multiline answer
        final_lines = []
        pipeline_found = False

        for line in logs:
            if line.strip().startswith("Aggregation Pipeline:"):
                pipeline_found = True
                continue
            if pipeline_found:
                final_lines.append(line)

        import re

        text = "\n".join(final_lines).strip()
        text = re.sub(r'\s*-\s*(\*\*[^*]+:\*\*)', r'\n- \1', text)
        text = re.sub(r'(:)\s*\n- ', r'\1\n\n- ', text)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()

        final_answer_md = text

        st.subheader("Final Answer")
        st.markdown(final_answer_md)
