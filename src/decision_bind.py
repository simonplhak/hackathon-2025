# decision_bind.py
import ast
import json
import re
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage

from requirement_creator import RequirementsList

# --- Simple, robust YES/NO router (no external module needed) ---
YES_PAT = re.compile(
    r"\b(yes|y|approve|approved|ok|okay|agree|accepted|accept|looks good|ship it)\b",
    re.I,
)
NO_PAT = re.compile(
    r"\b(no|n|reject|rejected|disagree|not ok|needs changes|question|questions|deny|denied)\b",
    re.I,
)


def route_on_user_decision(state: Dict[str, Any]) -> str:
    """
    Inspect the latest HumanMessage and return:
      - "approve" -> go to implementation
      - "reject"  -> go to question_maker
      - "unknown" -> ask again or stop (depending on your wiring)
    """
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    if not last_human:
        return "unknown"

    t = last_human.content.strip().lower()
    if YES_PAT.search(t):
        return "approve"
    if NO_PAT.search(t):
        return "reject"
    return "unknown"


# --- Nodes ---
def present_for_approval(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Show the last AI answer (JSON) to the user and ask for an approve/reject decision.
    """
    requirements: RequirementsList = RequirementsList.model_validate(
        state["messages"][-1].content[0]
    )
    # Use it
    json_display = format_json_as_bullets(requirements.model_dump_json())

    prompt = (
        "📝 **Please review the generated requirements below:**\n\n"
        f"{json_display}\n\n"
        "➡️ **Reply with one word:**\n"
        "  • 'approve' (or yes / ok / looks good) → continue to implementation\n"
        "  • 'reject' (or no / needs changes) → describe what to modify\n"
    )
    approval = input(prompt)

    return {
        **state,
        "messages": [HumanMessage(content=approval, name="present_for_approval")],
    }


def implementation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Approved path placeholder."""
    return {
        **state,
        "messages": state["messages"]
        + [AIMessage(content="✅ Approved. Proceeding to implementation…")],
    }


def format_json_as_bullets(raw: str) -> str:
    """Pretty-print payload as categories → bullet list. Accepts JSON or Python-literal strings."""

    def coerce_to_data(s: str):
        # Try real JSON first
        try:
            return json.loads(s)
        except Exception:
            pass
        # Then try Python literal (handles single-quoted dict/list)
        try:
            return ast.literal_eval(s)
        except Exception:
            return s  # give up

    data = coerce_to_data(raw)

    # If it's already structured as we expect: [{"requirements": [...]}]
    if (
        isinstance(data, list)
        and len(data) == 1
        and isinstance(data[0], dict)
        and "requirements" in data[0]
    ):
        data = data[0]

    # If it's a dict with "requirements": [...]
    if isinstance(data, dict) and isinstance(data.get("requirements"), list):
        groups: Dict[str, list] = {}
        for item in data["requirements"]:
            if not isinstance(item, dict):
                continue
            cat = (item.get("category") or "Uncategorized").title()
            desc = item.get("description") or ""
            if not desc:
                continue
            groups.setdefault(cat, []).append(desc)

        # Preferred category order
        preferred = ["Functional", "Non-Functional"]

        def sort_key(c):
            return (preferred.index(c) if c in preferred else 999, c)

        ordered_cats = sorted(groups.keys(), key=sort_key)

        # Build markdown
        lines = []
        for cat in ordered_cats:
            lines.append(f"### {cat}")
            seen = set()
            for desc in groups[cat]:
                if desc in seen:
                    continue
                seen.add(desc)
                lines.append(f"- {desc}")
            lines.append("")  # blank line after section
        return "\n".join(lines).rstrip()

    # Generic dict → bullets fallback
    if isinstance(data, dict):
        lines = []
        for k, v in data.items():
            lines.append(f"### {str(k).capitalize()}:")
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        item_str = ", ".join(
                            f"**{ik}**: {iv}" for ik, iv in item.items()
                        )
                        lines.append(f"- {item_str}")
                    else:
                        lines.append(f"- {item}")
            elif isinstance(v, dict):
                for ik, iv in v.items():
                    lines.append(f"- **{ik}**: {iv}")
            else:
                lines.append(f"- {v}")
            lines.append("")
        return "\n".join(lines).rstrip()

    # Fallback: pretty JSON-ish text if we can't structure it
    try:
        return "```json\n" + json.dumps(data, indent=2, ensure_ascii=False) + "\n```"
    except Exception:
        return str(data)
