# Define the Pydantic schema for a single requirement
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from typing import List
from langchain_core.prompts import ChatPromptTemplate
from retriever import RAGState
from langchain_core.messages import AIMessage


class Requirement(BaseModel):
    """A single software or business requirement extracted from a document."""

    requirement_id: str = Field(
        description="A unique ID for the requirement (e.g., 'REQ-001')."
    )
    description: str = Field(
        description="The detailed, testable description of the requirement."
    )
    category: str = Field(
        description="The requirement category (e.g., 'Functional', 'Non-Functional', 'User Story')."
    )


# Define the overall output schema for the list of requirements
class RequirementsList(BaseModel):
    """A list of extracted requirements."""

    requirements: List[Requirement] = Field(
        description="A list of all requirements found in the context."
    )


# Assuming llm is already defined as ChatGoogleGenerativeAI(model="gemini-1.5-flash")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


def generate_requirements(state: RAGState):
    """Node 2: Generate structured requirements from the retrieved documents."""
    print("--- GENERATING REQUIREMENTS ---")

    # 1. Instruct the LLM to use the Structured Output format
    # The .with_structured_output() method tells the LLM to format its response
    # as an object matching the RequirementsList schema.
    requirements_llm = llm.with_structured_output(RequirementsList)

    # 2. Define the Prompt for extraction
    extraction_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Requirements Engineer. "
                "Your task is to analyze the following CONTEXT and extract all possible "
                "software requirements, formatting them precisely according to the provided JSON schema. "
                "Ensure the output strictly follows the schema. Do not include any extra text or markdown."
                "\n\nCONTEXT:\n{context}",
            ),
            ("human", "Extract all requirements from the provided document context."),
        ]
    )

    # 3. Format the context from retrieved documents
    context = "\n---\n".join([doc.page_content for doc in state["documents"]])

    # 4. Invoke the chain to get the structured output
    structured_output = requirements_llm.invoke(
        extraction_prompt.format_messages(context=context)
    )

    # Convert the Pydantic model output to a user-friendly string to pass to the final agent
    requirements_str = structured_output.model_dump_json(indent=2)

    # Update state: Pass the extracted requirements as a new message for the final agent
    extracted_message = AIMessage(
        content=f"EXTRACTED REQUIREMENTS:\n\n{requirements_str}",
        name="requirements_agent",
    )

    return {"messages": [extracted_message]}
