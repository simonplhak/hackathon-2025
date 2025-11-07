"""
This module takes the amended requirements and generates a simple web application
implementation based on those requirements.
"""

from pathlib import Path
from typing import Dict, List
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from requirement_creator import RequirementsList
from retriever import RAGState

class WebAppFiles(BaseModel):
    """Generated web application files"""
    html_content: str = Field(description="Content of the index.html file")
    js_content: str = Field(description="Content of the main JavaScript file")

def extract_requirements_from_message(message: BaseMessage) -> RequirementsList | None:
    """
    Extracts requirements from an amender message
    Args:
        message: A message from the state that might contain requirements
    Returns:
        RequirementsList if found, None otherwise
    """
    if not (hasattr(message, 'name') and message.name == "requirements_amender"):
        return None
    
    return RequirementsList.model_validate(message.content[0])

def save_web_files(app_files: WebAppFiles, output_dir: Path) -> None:
    """
    Saves the generated web application files to the output directory
    Args:
        app_files: The generated file contents
        output_dir: Directory where to save the files
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save HTML file
    html_file = output_dir / "index.html"
    html_file.write_text(app_files.html_content)
    
    # Save JavaScript file
    js_file = output_dir / "main.js"
    js_file.write_text(app_files.js_content)

def implement_app(state: RAGState) -> Dict[str, List[BaseMessage]]:
    """
    LangChain node that generates a web application implementation based on requirements.
    Args:
        state: Current state containing messages with amended requirements
    Returns:
        Updated state with implementation details
    """
    print("\n--- IMPLEMENTING WEB APPLICATION ---")

    # 1. Find the most recent amended requirements in the message history
    requirements = None
    for msg in reversed(state["messages"]):
        requirements = extract_requirements_from_message(msg)
        if requirements:
            break

    if not requirements:
        return {
            "messages": [
                AIMessage(content="No amended requirements found to implement.")
            ]
        }

    # 2. Set up the LLM with structured output
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    implementation_llm = llm.with_structured_output(WebAppFiles)

    # 3. Create a prompt that instructs how to implement the web app
    implementation_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert Web Developer. Create a simple but modern web application "
            "that implements the following requirements. Generate both HTML and JavaScript "
            "code that satisfies these requirements while following best practices.\n\n"
            "Requirements:\n{requirements}\n\n"
            "Rules:\n"
            "1. Use modern HTML5 and JavaScript\n"
            "2. Create a simple minimalistic functional design\n"
            "3. Make sure the responsiveness works well and the javascript file is included properly\n"
            "4. Make sure all functionalities  and requirements are correctly implemented\n"
            "5. Be as brief as possible in your code generation while ensuring completeness\n"
            "6. The HTML file is named 'index.html' and the JavaScript file is named 'main.js'\n"
            "7. Do not use unnecessary styling but ensure the app is lined and structured well and the UI is usable."
            
        ),
        ("human", "Please generate the web application code based on these requirements.")
    ])

    # 4. Generate the implementation
    formatted_requirements = requirements.model_dump_json(indent=2)
    
    web_app = implementation_llm.invoke(
        implementation_prompt.format_messages(
            requirements=formatted_requirements
        )
    )

    # 5. Save the generated files
    output_dir = Path("out")
    save_web_files(web_app, output_dir)

    # 6. Return a message with the implementation details
    return {
        "messages": [
            AIMessage(
                content=[{
                    "message": "Web application has been generated",
                    "files": {
                        "html": str(output_dir / "index.html"),
                        "javascript": str(output_dir / "main.js")
                    }
                }],
                name="implementation"
            )
        ]
    }
