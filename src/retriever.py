from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import pymupdf4llm

load_dotenv()


# Define a custom state to include the retrieved documents
class RAGState(TypedDict):
    """
    Represents the state of our graph, including messages and retrieved documents.
    """

    messages: Annotated[List[BaseMessage], lambda x, y: x + y]  # Append messages
    documents: Annotated[List[Document], lambda x, y: x + y]  # Append documents


# --- RAG Components Setup (using Gemini Embeddings/LLM) ---
# NOTE: Replace 'os.getenv("GOOGLE_API_KEY")' with 'os.getenv("GEMINI_API_KEY")'
# if you are using the GEMINI key for both the LLM and Embeddings.


def get_retriever(pdf_path: Path):
    """Loads a PDF, creates a vector store, and returns a retriever."""
    if not pdf_path.exists():
        # NOTE: Using a placeholder path for example; ensure your file exists!
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    # 1. Load the PDF
    docs = pymupdf4llm.to_markdown(pdf_path)
    docs = Document(docs)

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents([docs])

    # 3. Create a Vector Store and Retriever
    # --- USE HUGGING FACE EMBEDDINGS (Local/Free) ---
    # The model 'all-MiniLM-L6-v2' is a small, fast, and good all-around model.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Creates an in-memory vector store (Chroma)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    return vectorstore.as_retriever()


# Initialize LLM (Ensure GEMINI_API_KEY is set in config.py or .env)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


def retrieve(state: RAGState, retriever):
    """Node 1: Retrieve relevant documents based on the last user message."""
    print("--- RETRIEVING DOCUMENTS ---")

    # Get the last user message
    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        # Only process if the last message is from the user
        return {"documents": []}

    # Use the retriever to fetch chunks related to the query
    docs = retriever.invoke(last_message.content)

    # Return the documents to be added to the state
    return {"documents": docs}
