# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# EMBEDDING SCRIPT CALLING
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# import sys
# import os

# # Add parent folder (src/) to path
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# from embedding_Class import RAGIndexer

# idx = RAGIndexer(
#     local_paths=["Dataset/Policies"],        # ✅ process files from this folder
#     s3_urls=None,                            # ✅ skip S3
#     workdir="rag_work",
#     embed_model="text-embedding-3-large"
# )

# idx.build()  # Extract → Chunk → Embed → Save index in RAM
# print("[DONE] Index ready with", len(idx.texts), "chunks")



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Retrival SCRIPT CALLING
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# #!/usr/bin/env python3
# """
# runner.py

# Runs the full flow:
# 1. Checks if embeddings already exist in RAM.
# 2. If not → calls RAGIndexer to build fresh embeddings.
# 3. Passes RAM index directly into Retriever.
# 4. Accepts query from terminal.
# 5. Performs retrieval + reranking.
# 6. Sends chunks to multimedia.py for final LLM answer.
# """

# import sys
# import traceback

# # --------------------------------------
# # Import embedding class
# # --------------------------------------
# try:
#     from embedding_Class import RAGIndexer
# except Exception as e:
#     print(f"[ERROR] Failed to import embedding_Class: {e}")
#     sys.exit(1)

# # --------------------------------------
# # Import retriever
# # --------------------------------------
# try:
#     from retrival_class import Retriever
# except Exception as e:
#     print(f"[ERROR] Failed to import retrival_class: {e}")
#     sys.exit(1)

# # --------------------------------------
# # Import multimedia.py
# # --------------------------------------
# try:
#     from multimedia import multimedia_response
# except Exception as e:
#     print(f"[ERROR] Failed to import multimedia.py: {e}")
#     multimedia_response = None


# # --------------------------------------
# # MAIN LOGIC
# # --------------------------------------
# def main():
#     print("\n==============================")
#     print("   Tata Play RAG - Runner")
#     print("==============================\n")

#     # Static local memory for this runner
#     # (Allows reuse without rebuilding)
#     global_runtime_cache = getattr(main, "_ram_cache", None)

#     # ---------------------------------------------------
#     # 1️⃣ CHECK IF EMBEDDINGS ALREADY PRESENT IN RAM
#     # ---------------------------------------------------
#     print("[STEP] Checking existing embeddings in RAM...")

#     if (
#         isinstance(global_runtime_cache, dict)
#         and "texts" in global_runtime_cache
#         and "vectors" in global_runtime_cache
#         and global_runtime_cache["texts"]
#         and global_runtime_cache["vectors"] is not None
#     ):
#         print("[INFO] ✅ Embeddings already present in RAM → Skipping embedding pipeline.")

#         texts = global_runtime_cache["texts"]
#         vectors = global_runtime_cache["vectors"]
#         metas = global_runtime_cache.get("metadatas", [{}] * len(texts))

#     else:
#         # ---------------------------------------------------
#         # 2️⃣ NO EMBEDDINGS FOUND → RUN EMBEDDING PIPELINE
#         # ---------------------------------------------------
#         print("[INFO] ❌ No embeddings in RAM → Running embedding pipeline now...")

#         idx = RAGIndexer(
#             local_paths=["Dataset/Policies"],   # ✅ Local policies directory
#             s3_urls=None,                       # ✅ No S3
#             workdir="rag_work",
#             embed_model="text-embedding-3-large"
#         )

#         idx.build()

#         if not idx.texts or idx.vectors is None:
#             print("[ERROR] Embedding index failed or returned no data.")
#             sys.exit(1)

#         print(f"[DONE] ✅ New index built with {len(idx.texts)} chunks.\n")

#         # Save to RAM cache for future queries
#         global_runtime_cache = {
#             "texts": idx.texts,
#             "vectors": idx.vectors,
#             "metadatas": idx.metadatas
#         }
#         main._ram_cache = global_runtime_cache

#         texts = idx.texts
#         vectors = idx.vectors
#         metas = idx.metadatas

#     # ---------------------------------------------------
#     # 3️⃣ INITIALIZE RETRIEVER
#     # ---------------------------------------------------
#     retriever = Retriever(
#         texts=texts,
#         metadatas=metas,
#         vectors=vectors
#     )

#     # ---------------------------------------------------
#     # 4️⃣ TAKE USER QUERY
#     # ---------------------------------------------------
#     query = input("Enter your query:\n> ").strip()
#     if not query:
#         print("Empty query. Exiting.")
#         return

#     # ---------------------------------------------------
#     # 5️⃣ RUN RETRIEVAL + RERANK
#     # ---------------------------------------------------
#     try:
#         print("\n[STEP] Performing retrieval...")

#         retrieval_output = retriever.retrieve(
#             query=query,
#             top_k=5,
#             rerank=True
#         )

#         if "error" in retrieval_output:
#             print(f"[ERROR] {retrieval_output['error']}")
#             return

#         candidates = retrieval_output.get("candidates", [])
#         context_chunks = [c["text"] for c in candidates]

#         print(f"[INFO] Retrieved {len(context_chunks)} chunks.\n")

#         # ---------------------------------------------------
#         # 6️⃣ CALL multimedia.py FOR FINAL ANSWER
#         # ---------------------------------------------------
#         if multimedia_response:
#             print("[STEP] Generating final response via multimedia.py...\n")
#             try:
#                 final_answer = multimedia_response(query, context_chunks)
#             except Exception as e:
#                 print("[ERROR] multimedia_response() failed:")
#                 print(e)
#                 print(traceback.format_exc())
#                 return

#             print("========= FINAL ANSWER =========\n")
#             print(final_answer)
#             print("================================\n")

#         else:
#             print("[WARN] multimedia_response() not found. Displaying raw chunks instead:")
#             for i, ch in enumerate(context_chunks):
#                 print(f"\n--- CHUNK {i+1} ---\n{ch[:800]}...\n")

#     except Exception as e:
#         print("[FATAL ERROR IN runner.py]")
#         print(e)
#         print(traceback.format_exc())


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
runner.py  —  MULTI-QUERY VERSION

✅ Builds embeddings only once
✅ Keeps them in RAM permanently
✅ Supports multiple queries in a single session
✅ Works in Streamlit or terminal
✅ Calls multimedia.py for final answer
"""

import sys
import traceback

# --------------------------------------
# Import embedding class
# --------------------------------------
try:
    from embedding_Class import RAGIndexer
except Exception as e:
    print(f"[ERROR] Failed to import embedding_Class: {e}")
    sys.exit(1)

# --------------------------------------
# Import retriever
# --------------------------------------
try:
    from retrival_class import Retriever
except Exception as e:
    print(f"[ERROR] Failed to import retrival_class: {e}")
    sys.exit(1)

# --------------------------------------
# Import multimedia.py
# --------------------------------------
try:
    from multimedia import multimedia_response
except Exception as e:
    print(f"[ERROR] Failed to import multimedia.py: {e}")
    multimedia_response = None


# --------------------------------------
# MAIN LOGIC
# --------------------------------------
def main():
    print("\n==============================")
    print("   Tata Play RAG - Runner")
    print("        MULTI QUERY MODE")
    print("==============================\n")

    # GLOBAL RAM CACHE for this script instance
    # Will persist until script is manually exited
    if not hasattr(main, "_ram_cache"):
        main._ram_cache = None

    ram = main._ram_cache

    # ---------------------------------------------------
    # 1️⃣ CHECK IF EMBEDDINGS ALREADY LOADED IN RAM
    # ---------------------------------------------------
    print("[STEP] Checking cached embeddings...")

    embeddings_ready = (
        isinstance(ram, dict)
        and "texts" in ram
        and "vectors" in ram
        and ram["texts"]
        and ram["vectors"] is not None
    )

    if embeddings_ready:
        print("[INFO] ✅ Embeddings already in RAM → No rebuild needed.")

        texts = ram["texts"]
        vectors = ram["vectors"]
        metas = ram.get("metadatas", [{}] * len(texts))

    else:
        # ---------------------------------------------------
        # 2️⃣ NO CACHED EMBEDDINGS → RUN INDEXER ONCE
        # ---------------------------------------------------
        print("[INFO] ❌ No embeddings found → Running RAGIndexer now...")

        idx = RAGIndexer(
            local_paths=["Dataset/Policies"],  # ✅ Local directory
            s3_urls=None,
            workdir="rag_work",
            embed_model="text-embedding-3-large"
        )

        idx.build()

        if not idx.texts or idx.vectors is None:
            print("[ERROR] Embedding pipeline failed. Cannot continue.")
            sys.exit(1)

        print(f"[DONE] ✅ Index built with {len(idx.texts)} chunks.\n")

        # ✅ Store entire dataset in RAM for multiple queries
        main._ram_cache = {
            "texts": idx.texts,
            "vectors": idx.vectors,
            "metadatas": idx.metadatas
        }

        texts = idx.texts
        vectors = idx.vectors
        metas = idx.metadatas

    # ---------------------------------------------------
    # 3️⃣ INITIALIZE RETRIEVER (re-used for all queries)
    # ---------------------------------------------------
    retriever = Retriever(
        texts=texts,
        metadatas=metas,
        vectors=vectors
    )

    print("\n✅ System ready. You can now ask unlimited questions.")
    print("Type 'exit' or 'quit' to stop.\n")

    # ---------------------------------------------------
    # 4️⃣ MULTI-QUERY LOOP
    # ---------------------------------------------------
    while True:
        query = input("Your Query:\n> ").strip()

        if query.lower() in ["exit", "quit", "close"]:
            print("\n✅ Exiting RAG system. Goodbye!\n")
            break

        if not query:
            print("⚠️ Empty query. Try again.\n")
            continue

        # ---------------------------------------------------
        # 5️⃣ RUN RETRIEVAL
        # ---------------------------------------------------
        try:
            print("\n[STEP] Retrieving relevant chunks...")

            retrieval_output = retriever.retrieve(
                query=query,
                top_k=5,
                rerank=True
            )

            if "error" in retrieval_output:
                print(f"[ERROR] {retrieval_output['error']}\n")
                continue

            candidates = retrieval_output.get("candidates", [])
            context_chunks = [c["text"] for c in candidates]

            print(f"[INFO] Retrieved {len(context_chunks)} chunks.\n")

            # ---------------------------------------------------
            # 6️⃣ CALL MULTIMEDIA FOR FINAL ANSWER
            # ---------------------------------------------------
            if multimedia_response:
                print("[STEP] Calling Final Answer Engine...\n")

                try:
                    final_answer = multimedia_response(query, context_chunks)
                except Exception as e:
                    print("[ERROR] multimedia_response failed:")
                    print(e)
                    print(traceback.format_exc())
                    continue

                print("========== FINAL ANSWER ==========\n")
                print(final_answer)
                print("\n==================================\n")

            else:
                print("[WARN] multimedia_response not found.")
                print("Showing retrieved chunks instead:\n")
                for i, ch in enumerate(context_chunks):
                    print(f"\n--- CHUNK {i+1} ---\n{ch[:500]}...\n")

        except Exception as e:
            print("\n[FATAL ERROR IN runner.py]\n")
            print(e)
            print(traceback.format_exc())
            continue


if __name__ == "__main__":
    main()
