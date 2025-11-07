from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from retriever import RAGState
from langchain_google_genai import ChatGoogleGenerativeAI


class CodeNotes(BaseModel):
    """Notes for improving code."""

    error_messages: str = Field(
        description="error message returned from syntax checker", default=""
    )
    user_comment: str = Field(description="Notes about the app by user review")
    code_review_suggestio: str = Field(
        description="Notes from reviewer about the code quality"
    )


def code_refactor(state: RAGState) -> dict:
    print("--- Refactoring code ---")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    structured_llm = llm.with_structured_output(None)  # TODO
    extraction_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Requirements Engineer. "
                "Your task is to analyze the following CONTEXT and come up with a questions clarifing the assigment"
                "software requirements, formatting them precisely according to the provided JSON schema. "
                "Ensure the output strictly follows the schema. Do not include any extra text or markdown."
                "\n\nCONTEXT:\n{context}",
            ),
            (
                "human",
                "What important do I need to ask my supervisor about the following requirements",
            ),
        ]
    )
