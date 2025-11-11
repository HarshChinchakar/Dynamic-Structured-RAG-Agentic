# import streamlit as st
# import traceback
# import os
# import sys
# import importlib
# import importlib.util

# # ------------------------------------------------
# # PATH SETUP
# # ------------------------------------------------
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# SRC_DIR = os.path.join(ROOT_DIR, "src")


# # ------------------------------------------------
# # SUPER-LOGGING MODULE LOADER
# # ------------------------------------------------
# def load_src_module(module_name: str):
#     st.markdown(f"### Loading Module: `{module_name}`")

#     full_name = f"src.{module_name}"

#     # Attempt 1: Normal import
#     try:
#         mod = importlib.import_module(full_name)
#         st.success(f"Imported via package: `{full_name}`")
#         return mod
#     except Exception as e:
#         st.warning(f"Normal import failed for `{full_name}`")
#         st.code(traceback.format_exc())

#     # Attempt 2: Fallback load from file
#     module_path = os.path.join(SRC_DIR, f"{module_name}.py")
#     st.write(f"Fallback loading from file: `{module_path}`")

#     if not os.path.isfile(module_path):
#         raise ImportError(f"Module file NOT found: {module_path}")

#     spec = importlib.util.spec_from_file_location(full_name, module_path)
#     mod = importlib.util.module_from_spec(spec)

#     sys.modules[full_name] = mod
#     sys.modules[module_name] = mod

#     try:
#         spec.loader.exec_module(mod)
#         st.success(f"Loaded successfully from file: `{module_path}`")

#         st.write("Module Attributes:")
#         st.json(sorted([x for x in dir(mod) if not x.startswith('_')]))

#         return mod
#     except Exception:
#         st.error(f"Exec failed for `{module_path}`")
#         st.code(traceback.format_exc())
#         raise


# # ------------------------------------------------
# # IMPORTS FOR TAB 1 (Policy RAG)
# # ------------------------------------------------
# try:
#     Router_mod = load_src_module("Router_gpt")
#     classify_query = getattr(Router_mod, "classify_query")
# except Exception:
#     st.error("Router import error:")
#     st.code(traceback.format_exc())
#     classify_query = None

# try:
#     Emb_mod = load_src_module("embedding_Class")
#     RAGIndexer = getattr(Emb_mod, "RAGIndexer")
#     st.success("RAGIndexer loaded.")
# except Exception:
#     st.error("Failed loading embedding_Class:")
#     st.code(traceback.format_exc())
#     st.stop()

# try:
#     Ret_mod = load_src_module("retrival_class")
#     Retriever = getattr(Ret_mod, "Retriever")
#     policy_handler_from_retriever = getattr(Ret_mod, "policy_handler_from_retriever", None)
#     st.success("Retriever loaded.")
# except Exception:
#     st.error("Failed loading retrival_class:")
#     st.code(traceback.format_exc())
#     st.stop()

# try:
#     Multi_mod = load_src_module("Mutlimedia")
#     multimedia_response = getattr(Multi_mod, "multimedia_response", None)
#     st.success("Mutlimedia loaded.")
# except Exception:
#     st.warning("Mutlimedia not loaded:")
#     st.code(traceback.format_exc())
#     multimedia_response = None


# # ------------------------------------------------
# # IMPORT FOR TAB 2 (Mongo HR Agent via uv)
# # ------------------------------------------------
# st.markdown("---")
# st.markdown("## Loading Mongo Document Agent (app.py)")

# try:
#     App_mod = load_src_module("app")

#     st.write("Attributes inside app.py:")
#     attrs = dir(App_mod)
#     st.json([x for x in attrs if not x.startswith("_")])

# except Exception:
#     st.error("Error loading app.py:")
#     st.code(traceback.format_exc())


# # ------------------------------------------------
# # PAGE CONFIG
# # ------------------------------------------------
# st.set_page_config(page_title="Policy + Mongo Agent", page_icon="🧠", layout="wide")
# st.title("🧠 AI Assistant — Policy RAG + Mongo Document Agent")


# # ------------------------------------------------
# # TABS
# # ------------------------------------------------
# tab1, tab2 = st.tabs(["📘 Policy RAG (Existing Debug Mode)", "🗄️ Document Query — Mongo Agent"])


# # ------------------------------------------------
# # ✅ TAB 1 — Policy RAG (UNCHANGED)
# # ------------------------------------------------
# with tab1:

#     # State init
#     if "rag_cache" not in st.session_state:
#         st.session_state.rag_cache = None

#     if "query_to_run" not in st.session_state:
#         st.session_state.query_to_run = None

#     # Inputs
#     user_query = st.text_area("Enter your question", height=150)
#     run = st.button("Run Query (Policy Only)")

#     rebuild = st.button("Rebuild Embeddings (force)")
#     if rebuild:
#         st.session_state.rag_cache = None
#         st.info("Cache cleared, embeddings will rebuild on next Run.")

#     POLICIES_PATH = os.path.join(ROOT_DIR, "Dataset", "Policies")

#     def build_index_debug():
#         st.write("Building index with FULL DEBUG...")

#         try:
#             idx = RAGIndexer(
#                 local_paths=[POLICIES_PATH],
#                 s3_urls=None,
#                 workdir="rag_work",
#                 embed_model="text-embedding-3-large",
#                 max_tokens=900,
#                 overlap=150,
#                 min_chunk_chars=280,
#             )

#             st.write("Calling idx.build() ...")
#             idx.build()

#             st.write("Texts extracted:", len(idx.texts))
#             st.write("Embeddings shape:", idx.vectors.shape if idx.vectors is not None else "None")
#             st.write("Sample metadata:", idx.metadatas[:3])

#             st.session_state.rag_cache = {
#                 "texts": idx.texts,
#                 "vectors": idx.vectors,
#                 "metadatas": idx.metadatas,
#                 "embed_model": idx.cfg.embed_model,
#             }

#             st.success("Embedding SUCCESS")

#         except Exception:
#             st.error("Embedding failed:")
#             st.code(traceback.format_exc())

#     if st.session_state.rag_cache is None:
#         build_index_debug()

#     if run:
#         if not user_query.strip():
#             st.warning("Enter a valid query.")
#             st.stop()

#         st.session_state.query_to_run = user_query.strip()

#     if st.session_state.query_to_run:

#         q = st.session_state.query_to_run

#         st.markdown("---")
#         st.header("DEBUG EXECUTION — POLICY ONLY")

#         cache = st.session_state.rag_cache

#         st.write("Creating retriever with cached embeddings...")
#         try:
#             retr = Retriever(
#                 texts=cache["texts"],
#                 vectors=cache["vectors"],
#                 metadatas=cache["metadatas"],
#                 embed_model=cache["embed_model"],
#             )
#         except Exception:
#             st.error("Retriever creation failed:")
#             st.code(traceback.format_exc())
#             st.stop()

#         st.write("Running retriever.retrieve() ...")
#         try:
#             ret = retr.retrieve(q, top_k=10, rerank=True)
#         except Exception:
#             st.error("Retriever failed:")
#             st.code(traceback.format_exc())
#             st.stop()

#         st.write("Retriever output (RAW):")
#         st.json(ret)

#         if "error" in ret:
#             st.error("Retriever returned error:", ret["error"])
#             st.stop()

#         candidates = ret.get("candidates", [])
#         chunks = [c["text"] for c in candidates]

#         # ✅ NEW — Chunk Expander
#         with st.expander("📄 Retrieved Chunks (Click to Expand)", expanded=False):
#             for i, c in enumerate(chunks):
#                 st.markdown(f"**Chunk {i+1}:**")
#                 st.code(c)

#         st.header("LLM ANSWER — DEBUG MODE")

#         try:
#             if multimedia_response:
#                 st.write("Using Mutlimedia.multimedia_response()")
#                 final_ans = multimedia_response(q, chunks)
#             else:
#                 st.write("Mutlimedia not available, fallback.")
#                 final_ans = "\n\n-----------\n\n".join(chunks)
#         except Exception:
#             st.error("LLM Answer generation failed:")
#             st.code(traceback.format_exc())
#             final_ans = f"[ERROR] {traceback.format_exc()}"

#         st.subheader("FINAL ANSWER")
#         st.write(final_ans)

#         st.session_state.query_to_run = None


# # ------------------------------------------------
# # ✅ TAB 2 — Mongo Agent (unchanged logic)
# # ------------------------------------------------
# with tab2:

#     st.header("Mongo HR Agent — Clean Mode")

#     email = st.text_input("User Email")
#     query = st.text_area("Document Query", height=150)

#     if st.button("Run Document Query"):

#         if not email.strip() or not query.strip():
#             st.warning("Please enter BOTH Email and Query.")
#             st.stop()

#         st.markdown("### Running Engine...")

#         import subprocess, shlex

#         uv_cmd = [
#             "uv", "run",
#             "src/app.py",
#             "--email", email,
#             "--query", query
#         ]

#         st.code(" ".join(shlex.quote(x) for x in uv_cmd))

#         logs = []

#         try:
#             proc = subprocess.Popen(
#                 uv_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.STDOUT,
#                 text=True,
#                 cwd=ROOT_DIR,
#                 bufsize=1
#             )
#         except Exception as e:
#             st.error("Failed to run uv.")
#             st.code(str(e))
#             st.stop()

#         for line in proc.stdout:
#             logs.append(line.rstrip("\n"))

#         proc.wait()

#         st.subheader("Execution Log")
#         st.code("\n".join(logs))

#         # ---------------------------------------------------
#         # Extract final answer (after Aggregation Pipeline)
#         # ---------------------------------------------------
#         final_lines = []
#         pipeline_found = False

#         for line in logs:
#             if line.strip().startswith("Aggregation Pipeline:"):
#                 pipeline_found = True
#                 continue
#             if pipeline_found:
#                 final_lines.append(line)

#         raw = "\n".join(final_lines).strip()

#         # Markdown cleanup
#         import re
#         text = raw
#         text = re.sub(r'\s*-\s*(\*\*[^*]+:\*\*)', r'\n- \1', text)
#         text = re.sub(r'(:)\s*\n- ', r'\1\n\n- ', text)
#         text = re.sub(r'\n{3,}', '\n\n', text).strip()

#         st.subheader("✅ Final Answer")
#         st.markdown(text)


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
# IMPORT ChatAgent for TAB 4
# ------------------------------------------------
try:
    ChatAgent_mod = load_src_module("chat_agent")   # expects src/chat_agent.py
    ChatAgent = getattr(ChatAgent_mod, "ChatAgent")
except Exception as e:
    st.error(f"Failed loading chat_agent.py: {e}")
    st.stop()

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
st.title("AI HR Assistant — Policy RAG + Mongo Document Agent")


# ------------------------------------------------
# TABS
# ------------------------------------------------
# tab1, tab2 = st.tabs(["📘 Policy RAG (Existing Debug Mode)", "🗄️ Document Query — Mongo Agent"])
# ------------------------------------------------
# TABS (now 3 tabs)
# ------------------------------------------------
# tab1, tab2, tab3 = st.tabs([
#     "📘 Policy RAG (Existing Debug Mode)",
#     "🗄️ Document Query — Mongo Agent",
#     "🔀 Router — Auto route"
# ])
tab1, tab2, tab3, tab4 = st.tabs([
    "📘 Policy RAG (Existing Debug Mode)",
    "🗄️ Document Query — Mongo Agent",
    "🔀 Router — Auto route",
    "💬 Policy Chat (With Memory)"
])



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

    # POLICIES_PATH = os.path.join(ROOT_DIR, "Dataset", "Policies")
    # CHange to above one for all file access - Or Update the below filename for specific file access
    POLICIES_FILE = os.path.join(ROOT_DIR, "Dataset", "Policies", "Policies.pdf")


    # FUNCTION
    def build_index_debug():
        try:
            idx = RAGIndexer(
                # local_paths=[POLICIES_PATH],
                local_paths=[POLICIES_FILE],
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

        with st.expander("Retrieved Result JSON (Click to Expand)", expanded=False):
            st.json(ret)


        if "error" in ret:
            st.error("Retriever returned error:", ret["error"])
            st.stop()

        candidates = ret.get("candidates", [])
        chunks = [c["text"] for c in candidates]

        with st.expander("Retrieved Chunks (Click to Expand)", expanded=False):
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


# # ------------------------------------------------
# # ✅ TAB 2 — Mongo Agent (Clean Mode)
# # ------------------------------------------------
# with tab2:

#     st.header("🗄️ Mongo HR Agent — Clean Mode")

#     email = st.text_input("User Email")
#     query = st.text_area("Document Query", height=150)

#     if st.button("Run Document Query"):

#         if not email.strip() or not query.strip():
#             st.warning("Please enter BOTH Email and Query.")
#             st.stop()

#         import subprocess, shlex

#         uv_cmd = [
#             "uv", "run",
#             "src/app.py",
#             "--email", email,
#             "--query", query
#         ]

#         logs = []

#         try:
#             proc = subprocess.Popen(
#                 uv_cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.STDOUT,
#                 text=True,
#                 cwd=ROOT_DIR,
#                 bufsize=1
#             )
#         except Exception as e:
#             st.error("Failed to run uv.")
#             st.code(str(e))
#             st.stop()

#         for line in proc.stdout:
#             logs.append(line.rstrip("\n"))

#         proc.wait()

#         st.subheader("Execution Log")
#         st.code("\n".join(logs))

#         # Extract final multiline answer
#         final_lines = []
#         pipeline_found = False

#         for line in logs:
#             if line.strip().startswith("Aggregation Pipeline:"):
#                 pipeline_found = True
#                 continue
#             if pipeline_found:
#                 final_lines.append(line)

#         import re

#         text = "\n".join(final_lines).strip()
#         text = re.sub(r'\s*-\s*(\*\*[^*]+:\*\*)', r'\n- \1', text)
#         text = re.sub(r'(:)\s*\n- ', r'\1\n\n- ', text)
#         text = re.sub(r'\n{3,}', '\n\n', text).strip()

#         final_answer_md = text

#         st.subheader("Final Answer")
#         st.markdown(final_answer_md)

# ------------------------------------------------
# ✅ TAB 2 — Mongo Agent (robust final-answer extraction)
# ------------------------------------------------
with tab2:

    st.header("🗄️ Mongo HR Agent — Clean Mode")

    email = st.text_input("User Email")
    query = st.text_area("Document Query", height=150)

    if st.button("Run Document Query"):

        if not email.strip() or not query.strip():
            st.warning("Please enter BOTH Email and Query.")
            st.stop()

        import subprocess, shlex, re

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

        # collect logs (no double printing)
        for line in proc.stdout:
            logs.append(line.rstrip("\n"))

        proc.wait()

        # Show full execution log once
        st.subheader("Execution Log")
        st.code("\n".join(logs))

        # -----------------------
        # Robust final-answer extraction
        # -----------------------
        final_lines = []
        last_structured_idx = -1

        # markers that indicate structured output or intermediate debug
        structured_patterns = [
            r'^Aggregation Pipeline:',   # aggregation marker
            r'^\{',                     # JSON / dict start
            r'^\[',                     # list start
            r'HumanMessage\(',          # langchain message
            r'AIMessage\(',             # langchain message
            r'^Fetched role',           # role log
            r'^Allowed$',               # agent decision line alone
            r'^Denied$',                # agent denied
        ]
        combined_re = re.compile("|".join(structured_patterns))

        # find last structured marker index
        for idx, line in enumerate(logs):
            if line is None:
                continue
            if combined_re.search(line):
                last_structured_idx = idx

        # everything after last_structured_idx is candidate final output
        if last_structured_idx + 1 < len(logs):
            final_lines = logs[last_structured_idx + 1 : ]
        else:
            final_lines = []

        # If final_lines empty, fallback: find last human-readable line
        if not final_lines:
            for line in reversed(logs):
                if not line:
                    continue
                # skip lines that clearly look structured or are small control words
                if combined_re.search(line):
                    continue
                # skip single-word control tokens
                if line.strip() in ("Allowed", "Denied"):
                    continue
                final_lines = [line]
                break

        # Compose raw final text
        raw = "\n".join(final_lines).strip()

        # Second-layer fallback: if still empty show last available line
        if not raw and logs:
            raw = logs[-1].strip()

        # Normalize possible inline bullets formatted like:
        # "... as follows: - **A:** ... - **B:** ..."
        if raw:
            # put bullets on their own line where pattern " - **" appears
            raw = re.sub(r'\s*-\s*(\*\*[^*]+:\*\*)', r'\n- \1', raw)
            # ensure blank line after header ending with colon, if bullets follow
            raw = re.sub(r'(:)\s*\n- ', r'\1\n\n- ', raw)
            # collapse excessive blank lines
            raw = re.sub(r'\n{3,}', '\n\n', raw).strip()

        # -----------------------
        # Display final answer
        # -----------------------
        st.subheader("Final Answer")
        if raw:
            # Render as markdown; preserves bullets and formatting
            st.markdown(raw)
        else:
            st.warning("Could not extract a final answer from logs.")
            if logs:
                st.code(logs[-1])

# ------------------------------------------------
# ✅ TAB 3 — Router (Policy or Document)
# ------------------------------------------------
with tab3:

    st.header("🔀 Auto Router — Policy / Document")

    router_query = st.text_area("Enter your query", height=140)
    router_email = st.text_input("User Email (required for Document queries)")

    if st.button("Run Routed Query"):

        if not router_query.strip():
            st.warning("Please enter a query.")
            st.stop()

        # -------------------------------
        # ✅ 1. CLASSIFY
        # -------------------------------
        st.subheader("Classification Result")

        try:
            cls_out = classify_query(router_query)
        except Exception:
            st.error("Router crashed:")
            st.code(traceback.format_exc())
            st.stop()

        # Router errors
        if isinstance(cls_out, dict) and "error" in cls_out:
            st.error("Router Error")
            st.json(cls_out)
            st.stop()

        # Extract router fields safely
        route, confidence, reason, doc_q, pol_q = cls_out
        route_value = route.value.lower()   # ✅ FIXED ENUM HANDLING

        st.write("**Route:**", route_value)
        st.write("**Confidence:**", confidence)
        st.write("**Reason:**", reason)

        # Freeze BOTH → always policy
        if route_value == "both":
            st.info("BOTH route detected → fallback to POLICY (BOTH is disabled).")
            route_value = "policy"
            router_query_to_run = pol_q or router_query
        else:
            router_query_to_run = router_query

        # -------------------------------
        # ✅ 2. POLICY FLOW
        # -------------------------------
        if route_value == "policy":

            st.subheader("Policy Flow (RAG + LLM)")

            # Ensure embeddings exist (exactly Tab1 logic)
            if st.session_state.get("rag_cache") is None:
                try:
                    idx = RAGIndexer(
                        # local_paths=[POLICIES_PATH],
                        local_paths=[POLICIES_FILE],
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
                except Exception:
                    st.error("Failed to build embeddings:")
                    st.code(traceback.format_exc())
                    st.stop()

            cache = st.session_state.rag_cache

            # Build retriever
            try:
                retr_local = Retriever(
                    texts=cache["texts"],
                    vectors=cache["vectors"],
                    metadatas=cache["metadatas"],
                    embed_model=cache["embed_model"],
                )
            except Exception:
                st.error("Retriever creation failed:")
                st.code(traceback.format_exc())
                st.stop()

            # Retrieve chunks (same as Tab1)
            try:
                q_to_run = pol_q or router_query
                retp = retr_local.retrieve(q_to_run, top_k=10, rerank=True)
            except Exception:
                st.error("Retriever failed:")
                st.code(traceback.format_exc())
                st.stop()

            # Retrieved JSON collapsed
            with st.expander("Retrieved Result JSON", expanded=False):
                st.json(retp)

            # Chunk display collapsed
            candidates = retp.get("candidates", [])
            chunks = [c["text"] for c in candidates]

            with st.expander("Retrieved Chunks", expanded=False):
                for i, c in enumerate(chunks):
                    st.markdown(f"### Chunk {i+1}")
                    st.code(c)

            # LLM final answer (same as Tab1)
            try:
                if multimedia_response:
                    final_ans = multimedia_response(q_to_run, chunks)
                else:
                    final_ans = "\n\n-----------\n\n".join(chunks)
            except Exception:
                st.error("LLM failed:")
                st.code(traceback.format_exc())
                final_ans = "[ERROR]"

            st.subheader("✅ Final Answer")
            st.write(final_ans)

        # -------------------------------
        # ✅ 3. DOCUMENT FLOW
        # -------------------------------
        elif route_value == "document":

            st.subheader("Document Flow (Mongo Agent)")

            if not router_email.strip():
                st.warning("Email is required for document queries.")
                st.stop()

            final_doc_query = doc_q or router_query

            import subprocess, shlex

            uv_cmd_doc = [
                "uv", "run",
                "src/app.py",
                "--email", router_email,
                "--query", final_doc_query
            ]

            logs_doc = []

            try:
                proc_doc = subprocess.Popen(
                    uv_cmd_doc,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=ROOT_DIR,
                    bufsize=1
                )
            except Exception as e:
                st.error("Failed starting document engine.")
                st.code(str(e))
                st.stop()

            for line in proc_doc.stdout:
                logs_doc.append(line.rstrip("\n"))
            proc_doc.wait()

            st.subheader("Execution Log")
            st.code("\n".join(logs_doc))

            # -------------------------------
            # ✅ Extract final answer (same as Tab 2)
            # -------------------------------
            import re

            combined_re = re.compile(
                r"^Aggregation Pipeline:|^\{|^\[|HumanMessage|AIMessage|^Fetched role|^Allowed$|^Denied$"
            )

            last_marker = -1
            for i, line in enumerate(logs_doc):
                if combined_re.search(line):
                    last_marker = i

            if last_marker + 1 < len(logs_doc):
                final_lines = logs_doc[last_marker + 1:]
            else:
                final_lines = []

            if not final_lines:
                for line in reversed(logs_doc):
                    if not line.strip():
                        continue
                    if not combined_re.search(line):
                        final_lines = [line]
                        break

            final_raw = "\n".join(final_lines).strip()

            # Format bullets if inline
            final_raw = re.sub(r'\s*-\s*(\*\*[^*]+:\*\*)', r'\n- \1', final_raw)
            final_raw = re.sub(r'(:)\s*\n- ', r'\1\n\n- ', final_raw)
            final_raw = re.sub(r'\n{3,}', '\n\n', final_raw).strip()

            st.subheader("✅ Final Answer")
            st.markdown(final_raw)

        else:
            st.error(f"Unknown route: {route_value}")
            st.stop()

# ------------------------------------------------
# ✅ TAB 4 — Policy Chat (Memory + RAG)
# ------------------------------------------------
with tab4:
    st.header("💬 Policy Chat — With Memory")

    # ---- Initialize agent only once ----
    if "chat_agent" not in st.session_state:
        if st.session_state.get("rag_cache") is None:
            st.error("Embeddings not built yet. Please run Tab 1 first.")
            st.stop()

        st.session_state.chat_agent = ChatAgent(st.session_state.rag_cache)

    # ---- Initialize UI chat history ----
    if "tab4_history" not in st.session_state:
        st.session_state.tab4_history = []

    # ---- Display existing conversation ----
    for msg in st.session_state.tab4_history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")

    st.markdown("---")

    # ---- Input box ----
    user_msg = st.text_input("Your message:", key="tab4_input")

    # ---- Submit ----
    if st.button("Send", key="tab4_send"):
        if not user_msg.strip():
            st.warning("Enter a message before sending.")
            st.stop()

        # Call our agent
        reply, updated_history, debug = st.session_state.chat_agent.chat_turn(user_msg)

        # Update UI history
        st.session_state.tab4_history = updated_history

        # Force rerun to update chat visually
        st.rerun()

    # ---- Reset Conversation ----
    if st.button("Reset Chat", key="tab4_reset"):
        st.session_state.chat_agent.clear()
        st.session_state.tab4_history = []
        st.success("Chat reset.")
        st.rerun()

