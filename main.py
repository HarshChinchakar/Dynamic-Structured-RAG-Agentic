
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
# main.py — Streamlit front-end (robust import for src modules)
# ================================================
# main.py — FULL DEBUG STREAMLIT FRONTEND
# Policy-only mode, prints EVERYTHING for debugging
# ================================================

# import streamlit as st
# from datetime import datetime, timezone
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

# st.write("✅ ROOT_DIR:", ROOT_DIR)
# st.write("✅ SRC_DIR:", SRC_DIR)


# # ------------------------------------------------
# # MODULE LOADER (prints EVERYTHING)
# # ------------------------------------------------
# def load_src_module(module_name: str):
#     st.write(f"🔍 Attempting to import module: {module_name}")

#     full_name = f"src.{module_name}"

#     # 1) normal import attempt
#     try:
#         mod = importlib.import_module(full_name)
#         st.success(f"✅ Imported via package: {full_name}")
#         return mod
#     except Exception as e:
#         st.warning(f"⚠️ Normal import failed for {full_name}: {e}")

#     # 2) fallback file import
#     module_path = os.path.join(SRC_DIR, f"{module_name}.py")
#     st.write("🔍 Fallback loading from file:", module_path)

#     if not os.path.isfile(module_path):
#         raise ImportError(f"❌ Module file not found: {module_path}")

#     spec = importlib.util.spec_from_file_location(full_name, module_path)
#     if spec is None or spec.loader is None:
#         raise ImportError("❌ Could not load import spec")

#     mod = importlib.util.module_from_spec(spec)
#     sys.modules[full_name] = mod
#     sys.modules[module_name] = mod  # convenience

#     try:
#         spec.loader.exec_module(mod)
#         st.success(f"✅ Loaded successfully from file: {module_path}")
#         return mod
#     except Exception as e:
#         st.error(f"❌ Exec failed for {module_path}: {e}")
#         raise


# # ------------------------------------------------
# # IMPORT MODULES WITH DEBUG LOGGING
# # ------------------------------------------------

# # 1) Router (we ignore its output; we hardcode POLICIES)
# try:
#     Router_mod = load_src_module("Router_gpt")
#     classify_query = getattr(Router_mod, "classify_query")
# except Exception as e:
#     st.error(f"Router import error: {e}")
#     classify_query = None

# # 2) Embedding class
# try:
#     Emb_mod = load_src_module("embedding_Class")
#     RAGIndexer = getattr(Emb_mod, "RAGIndexer")
#     st.success("✅ RAGIndexer loaded.")
# except Exception as e:
#     st.error(f"Failed loading embedding_Class: {e}")
#     st.stop()

# # 3) Retriever
# try:
#     Ret_mod = load_src_module("retrival_class")
#     Retriever = getattr(Ret_mod, "Retriever")
#     policy_handler_from_retriever = getattr(Ret_mod, "policy_handler_from_retriever", None)
#     st.success("✅ Retriever loaded.")
# except Exception as e:
#     st.error(f"Failed loading retrival_class: {e}")
#     st.stop()

# # 4) ✅ **Correct multimedia filename**
# try:
#     Multi_mod = load_src_module("Mutlimedia")     # ✅ FIXED — exactly as you said
#     multimedia_response = getattr(Multi_mod, "multimedia_response", None)
#     st.success("✅ Mutlimedia loaded.")
# except Exception as e:
#     st.warning(f"⚠️ Mutlimedia not loaded: {e}")
#     multimedia_response = None


# # ------------------------------------------------
# # STREAMLIT PAGE CONFIG
# # ------------------------------------------------
# st.set_page_config(page_title="Policy RAG — DEBUG", page_icon="🪵", layout="wide")
# st.title("🪵 FULL DEBUG — Policy RAG (Policy Only Mode)")


# # ------------------------------------------------
# # STATE INIT
# # ------------------------------------------------
# if "rag_cache" not in st.session_state:
#     st.session_state.rag_cache = None

# if "query_to_run" not in st.session_state:
#     st.session_state.query_to_run = None


# # ------------------------------------------------
# # UI INPUTS
# # ------------------------------------------------
# user_query = st.text_area("Enter your question", height=150)
# run = st.button("Run Query (Policy Only)")

# rebuild = st.button("Rebuild Embeddings (force)")
# if rebuild:
#     st.session_state.rag_cache = None
#     st.info("✅ Cache cleared, embeddings will rebuild on next Run.")


# # ------------------------------------------------
# # EMBEDDING LOGIC (DEBUG MODE)
# # ------------------------------------------------
# POLICIES_PATH = os.path.join(ROOT_DIR, "Dataset", "Policies")
# st.write("📁 Policy Directory:", POLICIES_PATH)


# def build_index_debug():
#     st.write("🔥 Building index with FULL DEBUG...")

#     try:
#         idx = RAGIndexer(
#             local_paths=[POLICIES_PATH],
#             s3_urls=None,
#             workdir="rag_work",
#             embed_model="text-embedding-3-large",
#             max_tokens=900,
#             overlap=150,
#             min_chunk_chars=280,
#         )

#         st.write("📌 Calling idx.build() ... watch logs below 👇")
#         idx.build()

#         # ✅ Print extracted texts count
#         st.write("✅ Texts extracted:", len(idx.texts))
#         st.write("✅ Embeddings shape:", idx.vectors.shape if idx.vectors is not None else "None")
#         st.write("✅ Sample metadata:", idx.metadatas[:3])

#         st.session_state.rag_cache = {
#             "texts": idx.texts,
#             "vectors": idx.vectors,
#             "metadatas": idx.metadatas,
#             "embed_model": idx.cfg.embed_model,
#         }

#         st.success("✅ Embedding SUCCESS — stored to RAM")

#     except Exception as e:
#         st.error("❌ Embedding failed:")
#         st.code(traceback.format_exc())


# if st.session_state.rag_cache is None:
#     build_index_debug()


# # ------------------------------------------------
# # QUERY EXECUTION — POLICY ONLY
# # ------------------------------------------------
# if run:
#     if not user_query.strip():
#         st.warning("Enter a valid query.")
#         st.stop()

#     st.session_state.query_to_run = user_query.strip()


# if st.session_state.query_to_run:
#     q = st.session_state.query_to_run

#     st.markdown("---")
#     st.header("🔎 DEBUG EXECUTION — POLICY ONLY")

#     # ------------------------------------------------
#     # 1) Retrieve chunks
#     # ------------------------------------------------
#     cache = st.session_state.rag_cache

#     st.write("🧠 Creating retriever with cached embeddings...")
#     try:
#         retr = Retriever(
#             texts=cache["texts"],
#             vectors=cache["vectors"],
#             metadatas=cache["metadatas"],
#             embed_model=cache["embed_model"],
#         )
#     except Exception as e:
#         st.error("Retriever creation failed:")
#         st.code(traceback.format_exc())
#         st.stop()

#     st.write("📌 Running retriever.retrieve() ...")
#     try:
#         ret = retr.retrieve(q, top_k=10, rerank=True)
#     except Exception as e:
#         st.error("Retriever failed:")
#         st.code(traceback.format_exc())
#         st.stop()

#     st.write("✅ Retriever output (RAW):")
#     st.json(ret)

#     if "error" in ret:
#         st.error("Retriever returned error:", ret["error"])
#         st.stop()

#     candidates = ret.get("candidates", [])
#     chunks = [c["text"] for c in candidates]

#     st.subheader("📄 Retrieved Chunks (Top 10)")
#     for i, c in enumerate(chunks):
#         st.code(f"[Chunk {i+1}] {c[:800]}")


#     # ------------------------------------------------
#     # 2) FINAL ANSWER (MULTIMEDIA RESPONSE OR CONCAT)
#     # ------------------------------------------------
#     st.header("🧠 LLM ANSWER — DEBUG MODE")

#     try:
#         if multimedia_response:
#             st.write("📌 Using Mutlimedia.multimedia_response()")
#             final_ans = multimedia_response(q, chunks)
#         else:
#             st.write("⚠️ Mutlimedia not available, falling back to concatenation.")
#             final_ans = "\n\n-----------\n\n".join(chunks)
#     except Exception as e:
#         st.error("LLM Answer generation failed:")
#         st.code(traceback.format_exc())
#         final_ans = f"[ERROR] {e}"

#     st.subheader("✅ FINAL ANSWER")
#     st.write(final_ans)

#     # reset
#     st.session_state.query_to_run = None


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

st.write("✅ ROOT_DIR:", ROOT_DIR)
st.write("✅ SRC_DIR:", SRC_DIR)


# ------------------------------------------------
# SUPER-LOGGING MODULE LOADER
# ------------------------------------------------
def load_src_module(module_name: str):
    st.markdown(f"### 🔍 Loading Module: `{module_name}`")

    full_name = f"src.{module_name}"

    # 1. Normal package import attempt
    try:
        mod = importlib.import_module(full_name)
        st.success(f"✅ Imported via package: `{full_name}`")
        return mod
    except Exception as e:
        st.warning(f"⚠️ Normal import failed for `{full_name}`")
        st.code(traceback.format_exc())

    # 2. Fallback to raw file load
    module_path = os.path.join(SRC_DIR, f"{module_name}.py")
    st.write(f"📄 Fallback loading from file:\n`{module_path}`")

    if not os.path.isfile(module_path):
        raise ImportError(f"❌ Module file NOT found: {module_path}")

    spec = importlib.util.spec_from_file_location(full_name, module_path)
    mod = importlib.util.module_from_spec(spec)

    sys.modules[full_name] = mod
    sys.modules[module_name] = mod  # convenience for direct import

    try:
        spec.loader.exec_module(mod)
        st.success(f"✅ Loaded successfully from file: `{module_path}`")

        # ✅ EXTRA DEBUG: List module attributes
        st.write("📌 **Module Attributes:**")
        st.json(sorted([x for x in dir(mod) if not x.startswith('_')]))

        return mod
    except Exception as e:
        st.error(f"❌ Exec failed for `{module_path}`")
        st.code(traceback.format_exc())
        raise


# ------------------------------------------------
# IMPORTS FOR TAB 1 (Policy RAG) — UNCHANGED
# ------------------------------------------------
try:
    Router_mod = load_src_module("Router_gpt")
    classify_query = getattr(Router_mod, "classify_query")
except Exception as e:
    st.error(f"Router import error: {e}")
    classify_query = None

try:
    Emb_mod = load_src_module("embedding_Class")
    RAGIndexer = getattr(Emb_mod, "RAGIndexer")
    st.success("✅ RAGIndexer loaded.")
except Exception as e:
    st.error(f"Failed loading embedding_Class:")
    st.code(traceback.format_exc())
    st.stop()

try:
    Ret_mod = load_src_module("retrival_class")
    Retriever = getattr(Ret_mod, "Retriever")
    policy_handler_from_retriever = getattr(Ret_mod, "policy_handler_from_retriever", None)
    st.success("✅ Retriever loaded.")
except Exception as e:
    st.error(f"Failed loading retrival_class:")
    st.code(traceback.format_exc())
    st.stop()

try:
    Multi_mod = load_src_module("Mutlimedia")
    multimedia_response = getattr(Multi_mod, "multimedia_response", None)
    st.success("✅ Mutlimedia loaded.")
except Exception as e:
    st.warning(f"⚠️ Mutlimedia not loaded:")
    st.code(traceback.format_exc())
    multimedia_response = None


# ------------------------------------------------
# IMPORTS FOR TAB 2 — Mongo Document Agent
# ------------------------------------------------
st.markdown("---")
st.markdown("## 🧩 Loading Mongo Document Agent (app.py)")

run_document_query = None

try:
    App_mod = load_src_module("app")  # loads src/app.py

    st.write("📌 Checking attributes inside app.py…")
    attrs = dir(App_mod)
    st.json([x for x in attrs if not x.startswith("_")])

    if "run_document_query" in attrs:
        run_document_query = getattr(App_mod, "run_document_query")
        st.success("✅ Found `run_document_query()`")
    else:
        st.error("❌ `run_document_query` NOT FOUND in app.py")
except Exception as e:
    st.error("❌ Error loading app.py:")
    st.code(traceback.format_exc())


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
# ✅ TAB 1 — UNMODIFIED (DO NOT TOUCH)
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
        st.info("✅ Cache cleared, embeddings will rebuild on next Run.")

    POLICIES_PATH = os.path.join(ROOT_DIR, "Dataset", "Policies")
    st.write("📁 Policy Directory:", POLICIES_PATH)

    # FUNCTION
    def build_index_debug():
        st.write("🔥 Building index with FULL DEBUG...")

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

            st.write("📌 Calling idx.build() ...")
            idx.build()

            st.write("✅ Texts extracted:", len(idx.texts))
            st.write("✅ Embeddings shape:", idx.vectors.shape if idx.vectors is not None else "None")
            st.write("✅ Sample metadata:", idx.metadatas[:3])

            st.session_state.rag_cache = {
                "texts": idx.texts,
                "vectors": idx.vectors,
                "metadatas": idx.metadatas,
                "embed_model": idx.cfg.embed_model,
            }

            st.success("✅ Embedding SUCCESS — stored to RAM")

        except Exception as e:
            st.error("❌ Embedding failed:")
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
        st.header("🔎 DEBUG EXECUTION — POLICY ONLY")

        cache = st.session_state.rag_cache

        st.write("🧠 Creating retriever with cached embeddings...")
        try:
            retr = Retriever(
                texts=cache["texts"],
                vectors=cache["vectors"],
                metadatas=cache["metadatas"],
                embed_model=cache["embed_model"],
            )
        except Exception as e:
            st.error("Retriever creation failed:")
            st.code(traceback.format_exc())
            st.stop()

        st.write("📌 Running retriever.retrieve() ...")
        try:
            ret = retr.retrieve(q, top_k=10, rerank=True)
        except Exception as e:
            st.error("Retriever failed:")
            st.code(traceback.format_exc())
            st.stop()

        st.write("✅ Retriever output (RAW):")
        st.json(ret)

        if "error" in ret:
            st.error("Retriever returned error:", ret["error"])
            st.stop()

        candidates = ret.get("candidates", [])
        chunks = [c["text"] for c in candidates]

        st.subheader("📄 Retrieved Chunks (Top 10)")
        for i, c in enumerate(chunks):
            st.code(f"[Chunk {i+1}] {c[:800]}")

        st.header("🧠 LLM ANSWER — DEBUG MODE")

        try:
            if multimedia_response:
                st.write("📌 Using Mutlimedia.multimedia_response()")
                final_ans = multimedia_response(q, chunks)
            else:
                st.write("⚠️ Mutlimedia not available, fallback.")
                final_ans = "\n\n-----------\n\n".join(chunks)
        except Exception as e:
            st.error("LLM Answer generation failed:")
            st.code(traceback.format_exc())
            final_ans = f"[ERROR] {e}"

        st.subheader("✅ FINAL ANSWER")
        st.write(final_ans)

        st.session_state.query_to_run = None


# ------------------------------------------------
# ✅ TAB 2 — Mongo Agent (clean multiline-mode)
# ------------------------------------------------
with tab2:

    st.header("🗄️ Mongo HR Agent — Clean Mode")

    email = st.text_input("User Email")
    query = st.text_area("Document Query", height=150)

    if st.button("Run Document Query"):

        if not email.strip() or not query.strip():
            st.warning("Please enter BOTH Email and Query.")
            st.stop()

        st.markdown("### 🔧 Running Engine...")

        import subprocess, shlex

        uv_cmd = [
            "uv", "run",
            "src/app.py",
            "--email", email,
            "--query", query
        ]

        st.code(" ".join(shlex.quote(x) for x in uv_cmd))

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

        # ✅ Show logs only once
        st.subheader("📄 Execution Log")
        st.code("\n".join(logs))

        # ---------------------------------------------------
        # ✅ Extract FINAL MULTILINE ANSWER
        # ---------------------------------------------------
        # ---------------------------------------------------
        # ✅ Extract FINAL MULTILINE ANSWER
        # ---------------------------------------------------
        final_lines = []
        pipeline_found = False
        
        for line in logs:
            if line.strip().startswith("Aggregation Pipeline:"):
                pipeline_found = True
                continue
            if pipeline_found:
                final_lines.append(line)
        
        # Raw answer as emitted after the pipeline
        final_answer_raw = "\n".join(final_lines).strip()
        
        # --- Normalize into proper Markdown bullets when the model prints
        #     on one line like: "... as follows: - **A:** ... - **B:** ..."
        import re
        
        text = final_answer_raw
        
        # 1) put each bullet on its own line (before " - **Bold:**")
        text = re.sub(r'\s*-\s*(\*\*[^*]+:\*\*)', r'\n- \1', text)
        
        # 2) ensure a blank line after the header line ending with a colon
        #    e.g. "… as follows:" -> (blank line) -> "- **Status:** …"
        text = re.sub(r'(:)\s*\n- ', r'\1\n\n- ', text)
        
        # 3) collapse any duplicated blank lines (optional but tidy)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        
        final_answer_md = text
        
        # ✅ Display final answer once, in theme-aware style (no custom HTML box)
        st.subheader("✅ Final Answer")
        st.markdown(final_answer_md)
