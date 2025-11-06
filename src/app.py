from src.Mongo import NaturalLanguageToMQL

from langchain_core.messages import HumanMessage
from src.langgraph_sample import access_agent

def run_document_query(email: str, query: str):
    """
    Document-only pipeline.
    Takes user email and natural language query.
    Returns final Mongo results or an error message.
    """

    # Build agent state
    state = {
        "email": email,
        "designation": "",
        "department": "",
        "region": "",
        "question": "",
        "intent": "",
        "decision": "",
        "messages": [HumanMessage(content=query)],
        "modified_query": ""
    }

    # -------------------------
    # 1) Invoke LangGraph Agent
    # -------------------------
    try:
        result = access_agent.invoke(state)
    except Exception as e:
        return {"status": "error", "message": f"Agent failed: {e}"}

    # Debug prints (optional)
    print("\n✅ Agent Output:")
    print(result)

    # -------------------------
    # 2) Check Access Decision
    # -------------------------
    decision = result.get("decision", "Denied")

    if decision != "Allowed":
        return {
            "status": "denied",
            "decision": decision,
            "message": "Access Denied. Cannot execute the query."
        }

    # -------------------------
    # 3) MQL Conversion + Query Execution
    # -------------------------
    converter = NaturalLanguageToMQL()

    modified_query = result.get("modified_query") or result.get("question")

    try:
        converter.convert_to_mql_and_execute_query(modified_query)
        mongo_output = converter.print_results(return_output=True)
    except Exception as e:
        return {"status": "error", "message": f"MongoDB execution failed: {e}"}

    return {
        "status": "success",
        "decision": decision,
        "query_used": modified_query,
        "agent_output": result,
        "mongo_output": mongo_output,
    }


# -------------- Run Locally --------------
if __name__ == "__main__":
    email = "manoranjand@tataplay.com"
    query = "what is my manager's email id?"

    output = run_document_query(email, query)
    print("\n✅ FINAL OUTPUT:\n", output)
