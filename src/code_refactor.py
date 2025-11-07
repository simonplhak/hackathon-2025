from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from implement_app import WebAppFiles, save_web_files
from retriever import RAGState
from langchain_core.messages import AIMessage
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
    notes = state["messages"][-1].content
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    structured_llm = llm.with_structured_output(WebAppFiles)
    implementation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Web Developer. Refactor web application."
                "Make sure that you generate code that follows best practices.\n\n"
                "Rules:\n"
                "1. Use modern HTML5 and JavaScript\n"
                "2. Keep design simple\n"
                "3. Make sure the responsiveness works well and the javascript file is included properly\n"
                "4. Make sure all functionalities  and requirements are correctly implemented\n"
                "5. Be as brief as possible in your code generation while ensuring completeness\n"
                "6. The HTML file is named 'index.html' and the JavaScript file is named 'main.js'\n"
                "7. Do not use unnecessary styling but ensure the app is lined and structured well and the UI is usable.\n\n"
                "```js\n{js_content}\n```\n"
                "\n```html{html_content}\n```\n"
                "Notes: {notes}\n",
            ),
            (
                "human",
                "Help me refactor my simple web app implemented in html and js.",
            ),
        ]
    )
    output_dir = Path("out")
    js_path = output_dir / "main.js"
    html_path = output_dir / "main.js"
    with js_path.open() as f:
        js_content = f.read()
    with html_path.open() as f:
        html_content = f.read()
    web_app = structured_llm.invoke(
        implementation_prompt.format_messages(
            js_content=js_content, html_content=html_content, notes=notes
        )
    )

    save_web_files(web_app, output_dir)
    return {
        "messages": [
            AIMessage(
                content=[
                    {
                        "message": "Web application has been generated",
                        "files": {
                            "html": str(output_dir / "index.html"),
                            "javascript": str(output_dir / "main.js"),
                        },
                    }
                ],
                name="code_refactor",
            )
        ]
    }
