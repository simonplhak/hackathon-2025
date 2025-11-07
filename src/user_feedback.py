from config import REPO_DIR
from retriever import RAGState
import re
from langchain_core.messages import HumanMessage

YES_PAT = re.compile(
    r"\b(yes|y|approve|approved|ok|okay|agree|accepted|accept|looks good|ship it)\b",
    re.I,
)
NO_PAT = re.compile(
    r"\b(no|n|reject|rejected|disagree|not ok|needs changes|question|questions|deny|denied)\b",
    re.I,
)


def run_server():
    import subprocess
    import sys

    # Define the command using the current Python executable
    command = [sys.executable, "-m", "http.server", "8000"]

    # Start the server in a new, non-blocking process
    server_process = subprocess.Popen(command, cwd=str(REPO_DIR / "out"))

    print(f"HTTP Server started on port 8000 with PID: {server_process.pid}")
    print("Access it at: http://127.0.0.1:8000/")
    return server_process


def kill_server(server_process):
    print("Terminating the server...")
    server_process.terminate()

    # Wait for the process to fully close
    server_process.wait()


# --- Nodes ---
def user_feedback(state: RAGState):

    prompt = (
        "📝 **Does app look okay to you?:**\n\n"
        "➡️ **Reply with one word:**\n"
        "  • 'approve' (or yes / ok / looks good) → continue to implementation\n"
        "  • 'reject' (or no / needs changes) → describe what to modify\n"
    )
    process = run_server()
    approval = input(prompt)
    kill_server(process)
    return {
        **state,
        "messages": [HumanMessage(content=approval, name="present_for_approval")],
    }


def user_notes(state: RAGState):
    prompt = "📝 **What do you want to improve?:**\n\n"
    # TODO: run app
    notes = input(prompt)
    return {
        **state,
        "messages": [HumanMessage(content=notes, name="user_notes")],
    }
