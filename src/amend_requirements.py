"""
This module handles the amendment of requirements based on user feedback through Q&A pairs.
It takes the original requirements and updates them based on specific questions and answers.
"""

from typing import List, Dict, TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, BaseMessage
from requirement_creator import Requirement, RequirementsList
from retriever import RAGState  # Import the shared state type

# Define our data structures for Q&A
class QuestionAnswer(BaseModel):
    """Represents a single question and its answer about requirements"""
    question: str = Field(description="Question asked about the requirements")
    answer: str = Field(description="User's answer providing feedback")

class AmendmentInput(BaseModel):
    """Complete input needed for the amendment process"""
    requirements: RequirementsList = Field(description="Original requirements to be modified")
    feedback: List[QuestionAnswer] = Field(description="List of Q&A pairs for amendments")
    

def create_mock_qa_messages() -> List[Dict[str, str]]:
    """
    Creates example Q&A pairs for testing/demonstration purposes.
    In a real application, these would come from actual user interactions.
    """
    return [
        {
            "question": "Should the calculator have a preferred color scheme?",
            "answer": "Yes. The calculator needs to have all the colors of the LGBTQ+ flag."
        }
    ]

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
        # Look for messages specifically marked as Q&A interactions
        if hasattr(msg, 'name') and msg.name == "requirements_qa":
            try:
                # The content should be a dictionary with 'question' and 'answer' keys
                if isinstance(msg.content, dict) and 'question' in msg.content and 'answer' in msg.content:
                    qa_pairs.append({
                        'question': msg.content['question'],
                        'answer': msg.content['answer']
                    })
            except Exception as e:
                print(f"Warning: Could not process Q&A message: {e}")
                continue
    
    return qa_pairs

def extract_requirements_from_message(message: BaseMessage) -> RequirementsList | None:
    """
    Extracts requirements from a message that contains JSON requirements data.
    Args:
        message: A message from the state that might contain requirements
    Returns:
        RequirementsList if found, None otherwise
    """
    if not (hasattr(message, 'name') and message.name == "requirements_agent"):
        return None
    
    import json
    # Remove the header text and parse the JSON content
    content = message.content.replace("EXTRACTED REQUIREMENTS:\n\n", "")
    return RequirementsList.model_validate(json.loads(content))

def amend_requirements(state: RAGState) -> Dict[str, List[BaseMessage]]:
    """
    LangChain node that updates requirements based on Q&A feedback.
    Args:
        state: Current state containing messages with requirements and Q&A pairs
    Returns:
        Updated state with amended requirements
    """
    print("\n--- AMENDING REQUIREMENTS BASED ON FEEDBACK ---")

    # 1. Find the most recent requirements in the message history
    requirements = None
    for msg in reversed(state["messages"]):
        requirements = extract_requirements_from_message(msg)
        if requirements:
            break

    if not requirements:
        return {
            "messages": [
                AIMessage(content="No requirements found to amend.")
            ]
        }
        

    # 2. Get Q&A pairs (in a real system, these would come from the message history)
    # qa_pairs = create_mock_qa_messages()
    qa_pairs = extract_qa_from_messages(state["messages"])

    # 3. Structure the input for the LLM
    amendment_input = AmendmentInput(
        requirements=requirements,
        feedback=[QuestionAnswer(**qa) for qa in qa_pairs]
    )

    # 4. Set up the LLM with structured output
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    requirements_llm = llm.with_structured_output(RequirementsList)

    # 5. Create a prompt that instructs how to amend requirements
    amendment_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert Requirements Engineer. Analyze the original requirements "
            "and Q&A feedback below, then produce updated requirements that incorporate "
            "the feedback while maintaining requirement quality and clarity.\n\n"
            "Original Requirements:\n{requirements}\n\n"
            "Feedback from Q&A:\n{qa_pairs}\n\n"
            "Rules:\n"
            "1. Preserve existing requirement IDs when possible\n"
            "2. Add new requirements with new IDs as needed\n"
            "3. Update descriptions based on feedback\n"
            "4. Ensure all requirements remain clear and testable"
        ),
        ("human", "Please update the requirements based on the Q&A feedback provided.")
    ])

    # 6. Prepare the content for the prompt
    formatted_requirements = amendment_input.requirements.model_dump_json(indent=2)
    formatted_qa = "\n".join([
        f"Q: {qa['question']}\nA: {qa['answer']}"
        for qa in qa_pairs
    ])

    # 7. Generate the updated requirements
    updated_requirements = requirements_llm.invoke(
        amendment_prompt.format_messages(
            requirements=formatted_requirements,
            qa_pairs=formatted_qa
        )
    )

    # 8. Format the output as a message
    requirements_str = updated_requirements.model_dump_json(indent=2)
    
    return {
        "messages": [
            AIMessage(
                content=f"AMENDED REQUIREMENTS:\n\n{requirements_str}",
                name="requirements_amender"
            )
        ]
    }
