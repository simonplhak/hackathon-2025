from pydantic import BaseModel, Field
from requirement_creator import RequirementsList
from retriever import RAGState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


class Questions(BaseModel):
    """Questions clarifying the requirements"""

    questions: list[str] = Field(description="list of questions about the requirements")


class QnA(BaseModel):
    question: str = Field(description="question about the requirements")
    answer: str = Field(description="answer provided by user on the question")


class QuestionsAndAnswers(BaseModel):
    """Questions and answers clarifying the requirements"""

    questions_and_answers: list[QnA] = Field(
        description="list of questions and answers clarifying the assigment"
    )


def question_maker(state: RAGState) -> dict:
    """
    Node 3: Refine, classify, or summarize the structured requirements.
    """
    print("--- REFINING REQUIREMENTS ---")

    # 1. Get the output from the previous node (the last message)
    last_message = state["messages"][-1]
    print(last_message)

    # # Ensure the message is from the requirements agent and extract the JSON string

    # # Extract the JSON string part
    # json_str = last_message.content.split("EXTRACTED REQUIREMENTS:\n\n", 1)[-1]

    # try:
    #     # Parse the JSON back into a Python object (dictionary/list)
    # --- CUSTOM LOGIC GOES HERE ---
    # Example: Count the requirements and generate a summary
    requirements: RequirementsList = RequirementsList.model_validate(
        last_message.content[0]
    )

    requirements_llm = llm.with_structured_output(Questions)

    # 2. Define the Prompt for extraction
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

    # 4. Invoke the chain to get the structured output
    structured_output = requirements_llm.invoke(
        extraction_prompt.format_messages(context=requirements.model_dump_json())
    )
    print(structured_output)
    qnas = []
    for question in structured_output.questions[:1]:
        answer = input(question)
        qnas.append(QnA(question=question, answer=answer))
    qnas = QuestionsAndAnswers(questions_and_answers=qnas)
    # # 2. Prepare the output message
    final_message = AIMessage(
        content=[qnas.model_dump()],
        name="question_maker",
    )

    # # # 3. Update state with the new message
    # # # IMPORTANT: The state structure is defined to aggregate messages, so we return a list.
    return {"messages": [final_message]}
