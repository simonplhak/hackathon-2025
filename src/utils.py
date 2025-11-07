from langchain_core.messages import BaseMessage
from requirement_creator import RequirementsList


def extract_requirements_from_message(message: BaseMessage) -> RequirementsList | None:
    """
    Extracts requirements from a message that contains JSON requirements data.
    Args:
        message: A message from the state that might contain requirements
    Returns:
        RequirementsList if found, None otherwise
    """
    if message.name is None:
        return None
    if "requirements" in message.name:
        return RequirementsList.model_validate(message.content[0])
    return None


def extract_requirements_from_state(state) -> RequirementsList:
    requirements = None
    for msg in reversed(state["messages"]):
        requirements = extract_requirements_from_message(msg)
        if requirements:
            break
    assert requirements is not None
    return RequirementsList.model_validate(requirements)
