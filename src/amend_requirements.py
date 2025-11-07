"""
This module handles the amendment of requirements based on user feedback through Q&A pairs.
It takes the original requirements and updates them based on specific questions and answers.
"""

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict
from langchain_core.messages import BaseMessage
from question_maker import QuestionsAndAnswers
from requirement_creator import RequirementsList
from retriever import RAGState  # Import the shared state type


# class AmendmentInput(BaseModel):
#     """Complete input needed for the amendment process"""

#     requirements: RequirementsList = Field(
#         description="Original requirements to be modified"
#     )
#     feedback: QuestionsAndAnswers = Field(
#         description="List of Q&A pairs for amendments"
#     )


def extract_qa_from_messages(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """
    Extracts Q&A pairs from the message history.
    Looks for messages with the name 'requirements_qa' that contain Q&A data.

    Args:
        messages: List of messages from the state
    Returns:
        List of Q&A pairs as dictionaries
    """
    qa_pairs = []

    for msg in messages:
        qa_pairs.append(
            {
                "question": msg.content["question"],
                "answer": msg.content["answer"],
            }
        )
    return qa_pairs


def extract_requirements_from_message(message: BaseMessage) -> RequirementsList | None:
    """
    Extracts requirements from a message that contains JSON requirements data.
    Args:
        message: A message from the state that might contain requirements
    Returns:
        RequirementsList if found, None otherwise
    """
    if "requirements" in message.name:
        return RequirementsList.model_validate(message.content[0])
    return None


def amend_requirements(state: RAGState) -> Dict[str, List[BaseMessage]]:
    """
    LangChain node that updates requirements based on Q&A feedback.
    Args:
        state: Current state containing messages with requirements and Q&A pairs
    Returns:
        Updated state with amended requirements
    """
    print("\n--- AMENDING REQUIREMENTS BASED ON FEEDBACK ---")

    # 1. Get the output from the previous node (the last message)
    qnas = QuestionsAndAnswers.model_validate(state["messages"][-1].content[0])
    requirements: RequirementsList = RequirementsList.model_validate(
        state["messages"][-2].content[0]
    )
    # 1. Find the most recent requirements in the message history
    # requirements = None
    for msg in reversed(state["messages"]):
        requirements = extract_requirements_from_message(msg)
        if requirements:
            break

    # # 2. Get Q&A pairs (in a real system, these would come from the message history)
    # # qa_pairs = create_mock_qa_messages()

    # # 3. Structure the input for the LLM
    # amendment_input = AmendmentInput(requirements=requirements, feedback=qnas)

    # # 4. Set up the LLM with structured output
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    requirements_llm = llm.with_structured_output(RequirementsList)

    # # 5. Create a prompt that instructs how to amend requirements
    amendment_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Requirements Engineer. Analyze the original requirements "
                "and Q&A feedback below, then produce updated requirements that incorporate "
                "the feedback while maintaining requirement quality and clarity.\n\n"
                "Original Requirements:\n{requirements}\n\n"
                "Feedback from Q&A:\n{qa_pairs}\n\n"
                "Rules:\n"
                "Update descriptions based on feedback\n"
                "Ensure all requirements remain clear and testable",
            ),
            (
                "human",
                "Please update the requirements based on the Q&A feedback provided.",
            ),
        ]
    )

    # # 6. Prepare the content for the prompt
    formatted_requirements = requirements.model_dump_json(indent=2)
    formatted_qnas = qnas.model_dump_json(indent=2)

    # # 7. Generate the updated requirements
    updated_requirements = requirements_llm.invoke(
        amendment_prompt.format_messages(
            requirements=formatted_requirements, qa_pairs=formatted_qnas
        )
    )

    return {
        "messages": [
            AIMessage(
                content=[updated_requirements.model_dump()],
                name="requirements_amender",
            )
        ]
    }
