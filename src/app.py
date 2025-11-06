# from Mongo import NaturalLanguageToMQL

# from langchain_core.messages import HumanMessage
# from langgraph_sample import access_agent

# def run_document_query(email: str, query: str):
#     """
#     Document-only pipeline.
#     Takes user email and natural language query.
#     Returns final Mongo results or an error message.
#     """

#     # Build agent state
#     state = {
#         "email": email,
#         "designation": "",
#         "department": "",
#         "region": "",
#         "question": "",
#         "intent": "",
#         "decision": "",
#         "messages": [HumanMessage(content=query)],
#         "modified_query": ""
#     }

#     # -------------------------
#     # 1) Invoke LangGraph Agent
#     # -------------------------
#     try:
#         result = access_agent.invoke(state)
#     except Exception as e:
#         return {"status": "error", "message": f"Agent failed: {e}"}

#     # Debug prints (optional)
#     print("\n✅ Agent Output:")
#     print(result)

#     # -------------------------
#     # 2) Check Access Decision
#     # -------------------------
#     decision = result.get("decision", "Denied")

#     if decision != "Allowed":
#         return {
#             "status": "denied",
#             "decision": decision,
#             "message": "Access Denied. Cannot execute the query."
#         }

#     # -------------------------
#     # 3) MQL Conversion + Query Execution
#     # -------------------------
#     converter = NaturalLanguageToMQL()

#     modified_query = result.get("modified_query") or result.get("question")

#     try:
#         converter.convert_to_mql_and_execute_query(modified_query)
#         mongo_output = converter.print_results(return_output=True)
#     except Exception as e:
#         return {"status": "error", "message": f"MongoDB execution failed: {e}"}

#     return {
#         "status": "success",
#         "decision": decision,
#         "query_used": modified_query,
#         "agent_output": result,
#         "mongo_output": mongo_output,
#     }


# # -------------- Run Locally --------------
# if __name__ == "__main__":
#     email = "manoranjand@tataplay.com"
#     query = "what is my manager's email id?"

#     output = run_document_query(email, query)
#     print("\n✅ FINAL OUTPUT:\n", output)


#!/usr/bin/env python3
import os
import sys
import json
import argparse

# ------------------------------------------------------------
# ✅ 1. FIX PYTHONPATH so scripts work under uv run
# ------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))     # /project/src
PROJECT_ROOT = os.path.dirname(ROOT)                  # /project
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ------------------------------------------------------------
# ✅ 2. Imports after sys.path fix
# ------------------------------------------------------------
from Mongo import NaturalLanguageToMQL
from langchain_core.messages import HumanMessage
from langgraph_sample import access_agent


# ------------------------------------------------------------
# ✅ 3. The actual pipeline function
# ------------------------------------------------------------
def run_pipeline(email: str, query: str) -> dict:
    """
    Runs the full HR Mongo Pipeline and returns JSON-safe response.
    """

    # --- Build LangGraph agent state ---
    state = {
        "email": email,
        "designation": "",
        "department": "",
        "region": "",
        "question": "",
        "intent": "",
        "decision": "",
        "messages": [HumanMessage(content=query)],
        "modified_query": "",
    }

    # --- Invoke routing/desicion agent ---
    try:
        agent_output = access_agent.invoke(state)
    except Exception as e:
        return {"status": "error", "message": f"Agent failed: {e}"}

    decision = agent_output.get("decision", "Denied")

    if decision != "Allowed":
        return {
            "status": "denied",
            "decision": decision,
            "message": "Access Denied by the HR Decision Engine.",
            "agent_output": agent_output,
        }

    # --- Mongo conversion + pipeline execution ---
    converter = NaturalLanguageToMQL()
    final_query = agent_output.get("modified_query") or agent_output.get("question")

    try:
        converter.convert_to_mql_and_execute_query(final_query)
        mongo_output = converter.print_results(return_output=True)
    except Exception as e:
        return {"status": "error", "message": f"MongoDB execution failed: {e}"}

    # --- Final JSON output ---
    return {
        "status": "success",
        "decision": decision,
        "query_used": final_query,
        "agent_output": agent_output,
        "mongo_output": mongo_output,
    }


# ------------------------------------------------------------
# ✅ 4. CLI entrypoint (uv run src/app.py --email ... --query ...)
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="HR Mongo Document Query Processor")
    parser.add_argument("--email", type=str, required=False)
    parser.add_argument("--query", type=str, required=False)

    # ✅ fallback to environment variables
    args = parser.parse_args()
    email = args.email or os.getenv("EMAIL")
    query = args.query or os.getenv("NATURAL_LANGUAGE_QUERY")

    if not email or not query:
        print(json.dumps({"status": "error", "message": "Missing email or query"}))
        sys.exit(1)

    return email, query


# ------------------------------------------------------------
# ✅ 5. MAIN EXECUTION (UV MODE)
# ------------------------------------------------------------
# -------------- Run Locally / CLI --------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mongo Document Agent CLI")
    parser.add_argument("--email", type=str, required=True, help="User email")
    parser.add_argument("--query", type=str, required=True, help="Natural-language query")
    args = parser.parse_args()

    email = args.email
    query = args.query

    # ✅ Run pipeline
    result = run_document_query(email, query)

    # ✅ Extract ONLY final answer — prevents HumanMessage JSON errors
    final_answer = (
        result.get("mongo_output")
        or result.get("final_answer")
        or result.get("answer")
        or ""
    )

    # ✅ Fallback safe string
    if not final_answer:
        # Convert non-serializable classes to strings safely
        final_answer = str({k: str(v) for k, v in result.items()})

    # ✅ Output ONLY the text
    print(final_answer)
