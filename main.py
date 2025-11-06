
# # main.py — Streamlit front-end for Router -> (Mongo | Policy) handlers
# import streamlit as st
# from datetime import datetime, timezone
# import traceback
# import os
# import re

# from src.Router_gpt import classify_query, RouteType

# # Lazy import Mutlimedia to prevent heavy init on Streamlit load
# import importlib
# Mutlimedia = importlib.import_module("src.Mutlimedia")
# load_in_memory_vectorstore = Mutlimedia.load_in_memory_vectorstore
# get_cached_vectorstore = Mutlimedia.get_cached_vectorstore
# clear_cache = Mutlimedia.clear_cache
# cache_status = Mutlimedia.cache_status
# policy_handler = Mutlimedia.policy_handler

# from langchain_openai import OpenAIEmbeddings
# from PyPDF2 import PdfReader
# from pptx import Presentation
# from docx import Document
# from langchain.text_splitter import CharacterTextSplitter

# # ===========================================
# # PAGE CONFIG
# # ===========================================
# st.set_page_config(page_title="Mongo_RAG Streamlit", page_icon="📚", layout="wide")
# st.title("📚 Mongo_RAG - Streamlit Frontend")

# # Tabs for Upload and Query
# tab1, tab2 = st.tabs(["📂 Upload & Embed", "💬 Query & Route"])

# # ===========================================
# # HELPER — Process and Embed Files Safely
# # ===========================================
# def process_and_embed_files(uploaded_files):
#     """Extract text, split safely, and embed once into in-memory vectorstore."""
#     try:
#         if get_cached_vectorstore():
#             st.info("Existing in-memory Chroma DB found — clearing old cache.")
#             clear_cache()

#         all_texts = []
#         splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

#         for uploaded_file in uploaded_files:
#             file_ext = os.path.splitext(uploaded_file.name)[1].lower()
#             text = ""

#             if file_ext == ".pdf":
#                 reader = PdfReader(uploaded_file)
#                 for page in reader.pages:
#                     t = page.extract_text()
#                     if t:
#                         text += t + "\n"

#             elif file_ext == ".pptx":
#                 prs = Presentation(uploaded_file)
#                 for slide in prs.slides:
#                     for shape in slide.shapes:
#                         if hasattr(shape, "text"):
#                             text += shape.text + "\n"

#             elif file_ext == ".docx":
#                 doc = Document(uploaded_file)
#                 for para in doc.paragraphs:
#                     text += para.text + "\n"

#             # Clean text and avoid large single-blocks
#             text = re.sub(r"\s+", " ", text).strip()
#             if not text:
#                 continue

#             chunks = splitter.split_text(text)

#             # Enforce max chunk length for token safety
#             safe_chunks = [c[:1000] if len(c) > 1000 else c for c in chunks]
#             all_texts.extend(safe_chunks)

#         if not all_texts:
#             st.error("❌ No valid text found in uploaded files.")
#             return False

#         embeddings = OpenAIEmbeddings()
#         load_in_memory_vectorstore(embeddings, all_texts, collection_name="ram_store")
#         st.success(f"✅ Successfully embedded {len(all_texts)} text chunks in memory!")
#         return True

#     except Exception as e:
#         st.error(f"❌ Embedding process failed: {e}")
#         st.code(traceback.format_exc())
#         return False


# # ===========================================
# # TAB 1 — UPLOAD & EMBED
# # ===========================================
# with tab1:
#     st.markdown("""
#     ### 📁 Upload Policy Documents
#     Upload **PDF**, **DOCX**, or **PPTX** files here to embed them in-memory.
#     These embeddings will later be used for answering Policy-related queries.
#     """)

#     uploaded_files = st.file_uploader(
#         "Upload one or more files",
#         type=["pdf", "docx", "pptx"],
#         accept_multiple_files=True
#     )

#     if uploaded_files:
#         with st.spinner("🔄 Processing and embedding uploaded files..."):
#             success = process_and_embed_files(uploaded_files)
#         if success:
#             st.caption(f"🧮 Cache status: {cache_status()}")
#     if st.button("🧹 Clear Cache"):
#         msg = clear_cache()
#         st.info(msg)

# # ===========================================
# # TAB 2 — QUERY + ROUTING
# # ===========================================
# with tab2:
#     st.markdown("""
#     ### 🧠 Intelligent Query Router
#     Enter a natural-language query.  
#     The router will:
#     1. Classify it as **Policy**, **Document**, or **Both**
#     2. Route to appropriate handler(s)
#     3. Return detailed results + full trace logs
#     """)

#     query = st.text_input("Enter your question", placeholder="e.g. What is the leave encashment policy?")

#     col1, col2 = st.columns([3, 1])
#     with col2:
#         if st.button("Run Query"):
#             if not query or not query.strip():
#                 st.warning("⚠️ Please enter a question.")
#             else:
#                 st.session_state['run_time'] = datetime.now(timezone.utc).isoformat()
#                 st.session_state['query'] = query.strip()

#     if 'run_time' in st.session_state:
#         st.caption(f"Last run UTC: {st.session_state['run_time']}")

#     # ==============================
#     # EXECUTION PIPELINE
#     # ==============================
#     if 'query' in st.session_state and st.session_state['query']:
#         q = st.session_state['query']

#         st.divider()
#         st.subheader("🔍 Step 1 — Classifying Query")

#         with st.spinner("Classifying query via Router..."):
#             try:
#                 router_response = classify_query(q)
#             except Exception as e:
#                 st.error(f"❌ Router crashed: {e}")
#                 st.code(traceback.format_exc())
#                 st.stop()

#         # Handle router output
#         if isinstance(router_response, dict) and "error" in router_response:
#             st.error(f"❌ Router error at stage **{router_response.get('stage')}**")
#             st.code(router_response["error"])
#             st.stop()
#         else:
#             try:
#                 route, confidence, reason, doc_q, pol_q = router_response
#             except Exception as e:
#                 st.error(f"❌ Unexpected Router output format: {e}")
#                 st.write(router_response)
#                 st.stop()

#         with st.expander("Router Output (Raw)", expanded=False):
#             st.json({
#                 "route": getattr(route, "value", str(route)),
#                 "confidence": confidence,
#                 "reason": reason,
#                 "doc_query": doc_q,
#                 "policy_query": pol_q,
#             })

#         st.success(f"✅ Router classified as **{getattr(route, 'value', str(route)).upper()}** (confidence {confidence})")

#         result_text = ""
#         stage_logs = []

#         # ==============================
#         # DOCUMENT HANDLER
#         # ==============================
#         if getattr(route, "value", str(route)).lower() == "document":
#             st.info("🗂️ Running Mongo (Document) handler...")
#             stage_logs.append("Router → Document Handler (Mongo)")
#             try:
#                 from src.Mongo import query_mongo
#                 with st.spinner("Executing Mongo pipeline..."):
#                     result_text = query_mongo(q)
#                 if isinstance(result_text, str) and result_text.startswith("ERROR"):
#                     st.error(result_text)
#                 else:
#                     st.success("✅ Document handler executed successfully.")
#             except Exception as e:
#                 stage_logs.append(f"Mongo Handler Error: {e}")
#                 st.error(f"❌ Document handler failed: {e}")
#                 st.code(traceback.format_exc())

#         # ==============================
#         # POLICY HANDLER
#         # ==============================
#         elif getattr(route, "value", str(route)).lower() == "policy":
#             st.info("📜 Running Policy handler...")
#             stage_logs.append("Router → Policy Handler")
#             try:
#                 if not get_cached_vectorstore():
#                     st.warning("⚠️ No in-memory Chroma DB found — please upload and embed documents first.")
#                     st.stop()

#                 with st.spinner("Executing Policy RAG pipeline..."):
#                     result_text = policy_handler(q)

#                 if not result_text:
#                     st.warning("⚠️ No response generated by Policy handler.")
#                 elif isinstance(result_text, str) and result_text.startswith("ERROR"):
#                     st.error(result_text)
#                 else:
#                     st.success("✅ Policy handler executed successfully.")

#             except Exception as e:
#                 stage_logs.append(f"Policy Handler Error: {e}")
#                 st.error("❌ Policy handler failed:")
#                 st.code(traceback.format_exc())

#         # ==============================
#         # BOTH HANDLERS
#         # ==============================
#         elif getattr(route, "value", str(route)).lower() == "both":
#             st.info("🔄 Running both Mongo & Policy handlers...")
#             stage_logs.append("Router → Both (Document + Policy)")
#             try:
#                 from src.Mongo import query_mongo
#                 if not get_cached_vectorstore():
#                     st.warning("⚠️ No in-memory Chroma DB found — please upload and embed documents first.")
#                     st.stop()

#                 with st.spinner("Executing Document handler..."):
#                     res_doc = query_mongo(doc_q or q)
#                 with st.spinner("Executing Policy handler..."):
#                     res_pol = policy_handler(pol_q or q)

#                 if any(isinstance(x, str) and x.startswith("ERROR") for x in [res_doc, res_pol]):
#                     st.error("⚠️ One or both handlers reported an error.")
#                 else:
#                     st.success("✅ Both handlers executed successfully.")

#                 result_text = f"--- DOCUMENT RESULT ---\n{res_doc}\n\n--- POLICY RESULT ---\n{res_pol}"

#             except Exception as e:
#                 stage_logs.append(f"Both Handler Error: {e}")
#                 st.error("❌ Combined handler failure:")
#                 st.code(traceback.format_exc())

#         # ==============================
#         # UNKNOWN ROUTE
#         # ==============================
#         else:
#             st.warning(f"⚠️ Unknown route type: {route}")
#             stage_logs.append("Unknown Route")

#         # ==============================
#         # OUTPUT DISPLAY
#         # ==============================
#         st.divider()
#         st.subheader("🧾 Final Result")
#         st.code(result_text or "No output generated.", language="text")

#         st.subheader("📋 Execution Log")
#         for log_entry in stage_logs:
#             st.write("-", log_entry)

#         st.caption("✅ Process completed successfully — all stages ran in same thread.")

# # main.py — Streamlit front-end (single-tab query + email)
# import streamlit as st
# from datetime import datetime, timezone
# import traceback
# import os

# # Router (existing)
# from src.Router_gpt import classify_query, RouteType

# # Import local modules (embedding + retriever + multimedia)
# # They live under src/
# try:
#     from src.embedding_Class import RAGIndexer
# except Exception as e:
#     st.error(f"Failed to import embedding_Class: {e}")
#     st.stop()

# try:
#     from src.retrival_class import Retriever, policy_handler_from_retriever
# except Exception as e:
#     st.error(f"Failed to import retrival_class: {e}")
#     st.stop()

# # multimedia.py is optional — policy_handler_from_retriever already produces answers,
# # but we'll import multimedia_response if you want to pass chunks to it in future.
# try:
#     from src.multimedia import multimedia_response
# except Exception:
#     multimedia_response = None  # optional

# # --------------------------
# # Page config
# # --------------------------
# st.set_page_config(page_title="Mongo_RAG - Query", page_icon="📚", layout="wide")
# st.title("📚 Mongo_RAG — Query (Single Tab)")

# # --------------------------
# # Session-state helpers
# # --------------------------
# if "rag_cache" not in st.session_state:
#     st.session_state.rag_cache = None  # will hold {'texts','vectors','metadatas','embed_model'}

# if "last_embedded" not in st.session_state:
#     st.session_state.last_embedded = None

# if "last_run" not in st.session_state:
#     st.session_state.last_run = None

# # --------------------------
# # Controls: Email + Query + Actions
# # --------------------------
# with st.container():
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         email = st.text_input("User Email (for audit / routing)", value=st.session_state.get("user_email", ""))
#         query = st.text_area("Enter your question", height=120, placeholder="e.g. How do I find my leave balance?")
#     with col2:
#         st.write("")  # spacer
#         if st.button("Run Query"):
#             st.session_state.user_email = email.strip()
#             st.session_state.query_to_run = query.strip()
#             st.session_state.last_run = datetime.now(timezone.utc).isoformat()

#         if st.button("Rebuild embeddings (force)"):
#             # Clear stored cache and prompt rebuild on next request
#             st.session_state.rag_cache = None
#             st.info("Embeddings cache cleared — the next Run Query will rebuild embeddings.")

# # --------------------------
# # Ensure we have a (single) embedding index loaded in session
# # - If not present, build it synchronously now (using Dataset/Policies)
# # --------------------------
# def build_index_from_policies():
#     """Build RAG index from local Dataset/Policies and store in session state."""
#     try:
#         with st.spinner("Building embeddings from Dataset/Policies (this runs once per session)..."):
#             idx = RAGIndexer(
#                 local_paths=["Dataset/Policies"],
#                 s3_urls=None,
#                 workdir="rag_work",
#                 embed_model="text-embedding-3-large",
#                 max_tokens=900,
#                 overlap=150,
#                 min_chunk_chars=280,
#             )
#             idx.build()

#             if not idx.texts or idx.vectors is None:
#                 st.error("Embedding pipeline completed but returned no data. Check files in Dataset/Policies.")
#                 return False

#             # Save to session_state
#             st.session_state.rag_cache = {
#                 "texts": idx.texts,
#                 "vectors": idx.vectors,
#                 "metadatas": idx.metadatas,
#                 "embed_model": idx.cfg.embed_model if hasattr(idx, "cfg") else "text-embedding-3-large",
#             }
#             st.session_state.last_embedded = datetime.now(timezone.utc).isoformat()
#             st.success(f"Embeddings ready — {len(idx.texts)} chunks loaded.")
#             return True
#     except Exception as e:
#         st.error("Failed to build embeddings:")
#         st.code(traceback.format_exc())
#         return False

# # If no cache, build now (synchronous)
# if st.session_state.rag_cache is None:
#     build_index_from_policies()

# # --------------------------
# # Run the pipeline when user pressed Run Query
# # --------------------------
# if st.session_state.get("query_to_run"):
#     q = st.session_state["query_to_run"]
#     st.markdown("---")
#     st.subheader("🔎 Execution Trace")

#     st.write("User:", st.session_state.get("user_email", "N/A"))
#     st.write("Query submitted at (UTC):", st.session_state.get("last_run"))

#     # 1) Classify
#     with st.spinner("Classifying query..."):
#         try:
#             router_response = classify_query(q)
#         except Exception as e:
#             st.error(f"Router crashed: {e}")
#             st.code(traceback.format_exc())
#             st.stop()

#     # Validate router response format
#     if isinstance(router_response, dict) and "error" in router_response:
#         st.error(f"Router error: {router_response}")
#         st.stop()

#     try:
#         route, confidence, reason, doc_q, pol_q = router_response
#     except Exception:
#         st.error("Router returned unexpected format.")
#         st.write(router_response)
#         st.stop()

#     st.write("Router decision:", getattr(route, "value", str(route)))
#     st.write("Confidence:", confidence)
#     st.write("Reason:", reason)

#     result_text = ""
#     stage_logs = []

#     # Handle routes
#     route_name = getattr(route, "value", str(route)).lower()

#     if route_name == "document":
#         st.info("🗂 Document route selected — currently unavailable.")
#         stage_logs.append("Router → Document Handler (NOT IMPLEMENTED)")
#         result_text = "Document handler is currently unavailable."

#     elif route_name == "policy":
#         stage_logs.append("Router → Policy Handler")
#         st.info("📜 Policy handler — retrieving from in-memory index.")

#         # Ensure embeddings present
#         cache = st.session_state.rag_cache
#         if not cache:
#             st.error("No embeddings present. Please upload/persist or rebuild embeddings first.")
#             st.stop()

#         # Instantiate Retriever once per run (lightweight)
#         try:
#             retr = Retriever(
#                 texts=cache["texts"],
#                 metadatas=cache.get("metadatas", [{}] * len(cache["texts"])),
#                 vectors=cache["vectors"],
#                 embed_model=cache.get("embed_model"),
#             )
#         except Exception as e:
#             st.error(f"Failed to create Retriever: {e}")
#             st.code(traceback.format_exc())
#             st.stop()

#         # Use policy_handler_from_retriever for a safe pipeline (returns answer + chunks)
#         try:
#             with st.spinner("Running retrieval + (optional) rerank + answer generation..."):
#                 answer, context_chunks = policy_handler_from_retriever(retr, q, top_k=5, rerank=True)
#         except Exception as e:
#             st.error(f"Policy pipeline failed: {e}")
#             st.code(traceback.format_exc())
#             st.stop()

#         # Present the outputs
#         if isinstance(answer, str) and answer.startswith("ERROR"):
#             st.error(answer)
#         else:
#             st.success("✅ Policy handler executed.")
#             st.subheader("📄 Final Answer")
#             st.write(answer)

#             with st.expander("🔎 Retrieved Chunks (for debugging)", expanded=False):
#                 for i, c in enumerate(context_chunks):
#                     st.markdown(f"**Chunk #{i+1}** (first 600 chars):")
#                     st.code(c[:600], language="text")

#             result_text = answer

#     elif route_name == "both":
#         # Run policy; document is unavailable
#         stage_logs.append("Router → Both (Document + Policy)")
#         st.info("Running policy part (document handler is unavailable).")

#         cache = st.session_state.rag_cache
#         if not cache:
#             st.error("No embeddings present. Please rebuild.")
#             st.stop()

#         retr = Retriever(
#             texts=cache["texts"],
#             metadatas=cache.get("metadatas", [{}] * len(cache["texts"])),
#             vectors=cache["vectors"],
#             embed_model=cache.get("embed_model"),
#         )

#         try:
#             with st.spinner("Running policy retrieval..."):
#                 answer, context_chunks = policy_handler_from_retriever(retr, pol_q or q, top_k=5, rerank=True)
#         except Exception as e:
#             st.error(f"Policy part failed: {e}")
#             st.code(traceback.format_exc())
#             st.stop()

#         result_text = f"--- POLICY RESULT ---\n{answer}\n\n--- DOCUMENT RESULT ---\nCurrently unavailable."

#     else:
#         st.warning(f"Unknown route type: {route_name}")
#         stage_logs.append("Unknown Route")
#         result_text = "No handler for this route."

#     # Display final result and logs
#     st.markdown("---")
#     st.subheader("🧾 Final Result")
#     st.code(result_text or "No result produced.", language="text")

#     st.subheader("📋 Execution Log")
#     for entry in stage_logs:
#         st.write("-", entry)

#     # clear the saved query so the user can type a new one
#     del st.session_state["query_to_run"]

# ===========================================
# ✅ main.py — Streamlit Frontend for RAG System
# ===========================================

import streamlit as st
from datetime import datetime, timezone
import traceback
import os
import sys

# -----------------------------------------------------
# ✅ Ensure project root is importable inside Streamlit Cloud
# -----------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(ROOT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# -----------------------------------------------------
# ✅ Import Router
# -----------------------------------------------------
try:
    from src.Router_gpt import classify_query, RouteType
except Exception as e:
    st.error(f"❌ Failed to import Router: {e}")
    st.stop()

# -----------------------------------------------------
# ✅ Import Embedding, Retriever, Multimedia
# -----------------------------------------------------
try:
    from src.embedding_Class import RAGIndexer
    from src.retrival_class import Retriever
    from src.multimedia import multimedia_response
except Exception as e:
    st.error(f"❌ Failed to import core RAG modules: {e}")
    st.code(traceback.format_exc())
    st.stop()

# -----------------------------------------------------
# ✅ GLOBAL RAM CACHE (PERSIST for whole Streamlit session)
# -----------------------------------------------------
if "rag_cache" not in st.session_state:
    st.session_state.rag_cache = None


# ===========================================
# ✅ PAGE CONFIG
# ===========================================
st.set_page_config(page_title="Tata Play - Policy RAG", page_icon="📘", layout="wide")
st.title("📘 Tata Play HR Policy Assistant")


# ===========================================
# ✅ LOAD EMBEDDINGS (Runs Once)
# ===========================================
def load_policy_embeddings_once():

    if st.session_state.rag_cache:
        st.info("✅ Reusing existing RAM embeddings (already loaded).")
        return st.session_state.rag_cache

    st.info("📦 Building embeddings from Dataset/Policies (runs once per session)...")

    try:
        idx = RAGIndexer(
            local_paths=["Dataset/Policies"],   # ✅ Hard-coded local folder
            s3_urls=None,
            workdir="rag_work",
            embed_model="text-embedding-3-large"
        )

        idx.build()

        if not idx.texts or idx.vectors is None:
            st.error("❌ Embedding creation failed — no data extracted.")
            return None

        cache = {
            "texts": idx.texts,
            "vectors": idx.vectors,
            "metadatas": idx.metadatas
        }

        st.session_state.rag_cache = cache
        st.success(f"✅ Loaded {len(idx.texts)} chunks into RAM.")

        return cache

    except Exception as e:
        st.error("❌ Failed during embedding pipeline.")
        st.code(traceback.format_exc())
        return None


# ===========================================
# ✅ Streamlit UI
# ===========================================

st.subheader("🔑 User Details")
user_email = st.text_input("Enter your email:", placeholder="name@company.com")

st.divider()


st.subheader("💬 Ask a Policy or Document Question")
query = st.text_area("Your Query", height=120)


if st.button("Run Query"):
    if not user_email.strip():
        st.warning("⚠️ Please enter your email before running queries.")
        st.stop()

    if not query.strip():
        st.warning("⚠️ Enter a query to proceed.")
        st.stop()

    # Load embeddings into RAM
    cache = load_policy_embeddings_once()
    if cache is None:
        st.error("❌ Cannot answer queries without embeddings.")
        st.stop()

    # ----------------------------------------
    # ✅ STEP 1 — Router Classification
    # ----------------------------------------
    st.subheader("🔍 Step 1 — Router Classification")

    try:
        route, confidence, reason, doc_q, pol_q = classify_query(query)
    except Exception as e:
        st.error(f"❌ Router crashed: {e}")
        st.code(traceback.format_exc())
        st.stop()

    st.json({
        "route": route.value if hasattr(route, "value") else str(route),
        "confidence": confidence,
        "reason": reason,
        "doc_query": doc_q,
        "policy_query": pol_q,
    })

    st.success(f"✅ Classified as **{route.value.upper()}** (confidence {confidence})")
    st.divider()

    # ----------------------------------------
    # ✅ STEP 2 — Policy Handling
    # ----------------------------------------
    if route.value == "policy":
        st.subheader("📘 Step 2 — Policy Answer")

        retriever = Retriever(
            texts=cache["texts"],
            metadatas=cache["metadatas"],
            vectors=cache["vectors"]
        )

        with st.spinner("🔎 Retrieving relevant chunks..."):
            ret = retriever.retrieve(query=query, top_k=5, rerank=True)

        if "error" in ret:
            st.error(ret["error"])
            st.stop()

        chunks = [c["text"] for c in ret["candidates"]]

        with st.spinner("🧠 Generating final answer..."):
            answer = multimedia_response(query, chunks)

        st.success("✅ Final Answer")
        st.write(answer)
        st.divider()


    # ----------------------------------------
    # ✅ STEP 3 — Document Handling (NOT READY)
    # ----------------------------------------
    elif route.value == "document":
        st.subheader("🗂️ Document Handler")
        st.warning("📁 Document RAG is not implemented yet.")
        st.info("✅ Router worked — but document features are pending.")

    # ----------------------------------------
    # ✅ STEP 4 — Both
    # ----------------------------------------
    else:
        st.subheader("🔄 Combined (Document + Policy) Query")
        st.warning("⚠️ Document system is unavailable — showing only Policy result.")

        retriever = Retriever(
            texts=cache["texts"],
            metadatas=cache["metadatas"],
            vectors=cache["vectors"]
        )

        with st.spinner("Retrieving chunks..."):
            ret = retriever.retrieve(query=query, top_k=5, rerank=True)

        chunks = [c["text"] for c in ret["candidates"]]

        with st.spinner("Generating answer..."):
            answer = multimedia_response(query, chunks)

        st.success("✅ Final Answer (Policy only)")
        st.write(answer)

