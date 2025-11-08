from pathlib import Path
from typing import Dict, List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from retriever import RAGState


def _read_text_or_none(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


class RefactorCommentsList(BaseModel):
    comments: List[str] = Field(
        description="A list of all refactor comments to implementaion files."
    )


def refactor_comment(state: RAGState) -> Dict[str, List[BaseMessage]]:
    """
    Node: reads local web files and returns a single long refactor commentary string.
    """
    print("\n--- REFACTORING FROM LOCAL FILES ---")

    root = Path("out")
    html_path = root / "index.html"
    js_path = root / "main.js"

    html_src = _read_text_or_none(html_path)
    js_src = _read_text_or_none(js_path)

    if not html_src and not js_src:
        return {
            "messages": [
                AIMessage(
                    # Keep it a plain error message; main() will print this string.
                    content=(
                        "No inputs to review: neither out/index.html nor out/main.js "
                        "could be read. Generate the app first or place the files under ./out."
                    ),
                    name="refactor_commentary",
                )
            ]
        }

    # Provide placeholders so the prompt remains well-formed if one file is missing
    html_block = html_src if html_src is not None else "<missing index.html>"
    js_block = js_src if js_src is not None else "<missing main.js>"

    # 2. Set up the LLM with structured output
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    implementation_llm = llm.with_structured_output(RefactorCommentsList)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a principal front-end engineer. Read the provided HTML and "
                    "JavaScript and return a SINGLE, detailed refactoring commentary. Rules:\n"
                    "• Output ONLY the commentary text (no JSON/YAML; code blocks only when essential).\n"
                    "• Be specific and actionable (reference selectors, functions, or snippets).\n"
                    "• Cover: semantics & accessibility (landmarks, labels, roles, focus order),\n"
                    "  performance (loading strategy, caching, images, tree-shaking),\n"
                    "  architecture (components, state, events, separation of concerns),\n"
                    "  security (XSS, CSP, URL handling),\n"
                    "  UX (responsive layout, keyboard/ARIA, error/empty states),\n"
                    "  testing/tooling (unit/e2e, lint/format, CI), and provide concise improved\n"
                    "  snippets only when they clarify the change.\n"
                    "• Do NOT restate original code wholesale; focus on improvements.\n"
                ),
            ),
            (
                "human",
                (
                    "Review the following files and produce ONE long refactor commentary.\n\n"
                    f"=== {html_path} ===\n"
                    "```html\n{html}\n```\n\n"
                    f"=== {js_path} ===\n"
                    "```javascript\n{js}\n```"
                ),
            ),
        ]
    )

    messages = prompt.format_messages(html=html_block, js=js_block)
    res = implementation_llm.invoke(messages)
    if res is None:
        commentary_text = ""
    else:
        commentary_text = res.comments

    return {
        "messages": [
            AIMessage(
                content=commentary_text,
                name="refactor_commentary",
            )
        ]
    }
