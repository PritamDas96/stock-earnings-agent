"""The earnings-analysis agent as a compiled LangGraph.

Built on LangGraph's prebuilt ReAct agent: the model plans, calls tools, reads
their results and iterates until it can answer. An in-memory checkpointer gives
each ``thread_id`` its own conversation memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from agents.llm import get_llm
from agents.state import SYSTEM_PROMPT
from agents.tools import build_agent_tools
from core import get_logger

log = get_logger("agent")


def build_agent(model: str | None = None):
    """Compile the ReAct earnings agent graph.

    Args:
        model: Optional Groq model override.

    Returns:
        A compiled LangGraph runnable.
    """
    llm = get_llm(model=model)
    tools = build_agent_tools()
    return create_react_agent(
        llm,
        tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )


@dataclass
class EarningsAgent:
    """Convenience wrapper around the compiled agent graph.

    Example:
        >>> agent = EarningsAgent()
        >>> print(agent.analyze("How did AAPL's margins trend this year?"))
    """

    model: str | None = None
    _graph: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._graph = build_agent(self.model)

    def analyze(self, question: str, thread_id: str = "default", recursion_limit: int = 25) -> str:
        """Answer a question, returning the agent's final text response.

        Args:
            question: The user's analytical question.
            thread_id: Conversation id; reuse to continue a conversation.
            recursion_limit: Max reasoning/tool steps before stopping.
        """
        log.info("Agent analysing (thread={}): {}", thread_id, question)
        result = self._graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit},
        )
        messages = result.get("messages", [])
        for message in reversed(messages):
            if isinstance(message, AIMessage) and message.content:
                return message.content
        return "The agent did not produce a textual answer."

    def stream(self, question: str, thread_id: str = "default"):
        """Yield intermediate graph steps for live UIs.

        Yields:
            Per-node state updates from the LangGraph execution.
        """
        yield from self._graph.stream(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="updates",
        )
