from pathlib import Path
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from config import DATA_DIR

from requirement_creator import generate_requirements
from retriever import RAGState, get_retriever, retrieve

load_dotenv()


def main(path: Path):
    retriever = get_retriever(path)

    # 2. Build the Graph
    graph = StateGraph(RAGState)

    # Add nodes
    # Note: We use a lambda to pass the retriever to the retrieve node
    graph.add_node("retrieve", lambda state: retrieve(state, retriever))
    graph.add_node("generate_requirements", generate_requirements)

    # Define the workflow (always go from retrieve to generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate_requirements")
    graph.add_edge("generate_requirements", END)

    # Compile and run
    app = graph.compile()

    user_query = "What are the core requirements and key details in this document?"
    print(f"\nUser: {user_query}")

    # 2. Invoke the compiled graph ('app') with the initial state
    res = app.invoke(
        # The initial state must match the RAGState structure.
        {"messages": [HumanMessage(content=user_query)]}
    )

    # The final answer is the last message generated
    final_answer = res["messages"][-1].content
    print(f"\nAI Agent Final Response:\n{final_answer}")


if __name__ == "__main__":
    path = DATA_DIR / "00.pdf"
    main(path)
