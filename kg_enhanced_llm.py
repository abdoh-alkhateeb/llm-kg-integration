"""KG-enhanced LLM: answer questions grounded in a knowledge graph.

This module implements the *KG-enhanced LLM* application — the graph is used
as the sole source of truth and is serialized into the LLM context so that
answers are constrained to facts present in the store.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import MAX_QA_CONTEXT_CHARS, OLLAMA_BASE_URL, OLLAMA_MODEL
from kg_store import KnowledgeGraphStore

SYSTEM_PROMPT = """You answer questions using only the provided knowledge graph context.

Rules:
- Use only facts that appear in the knowledge graph.
- If the graph does not contain enough information, reply exactly: "The knowledge graph does not contain enough information to answer that."
- Keep answers concise and reference the entities or relationships involved.
"""

_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=SYSTEM_PROMPT),
        ("human", "Knowledge graph:\n{graph_context}\n\nQuestion:\n{question}"),
    ]
)

_LLM = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)

_QA_CHAIN = _PROMPT | _LLM | StrOutputParser()


class QuestionAnsweringError(RuntimeError):
    """Raised when the underlying LLM call fails."""


@dataclass(frozen=True)
class AnswerResult:
    """A QA answer together with the context the LLM was given."""

    answer: str
    context: str


def answer_question(store: KnowledgeGraphStore, question: str) -> AnswerResult:
    """Answer `question` using only facts in `store`."""
    question = question.strip()
    if not question:
        return AnswerResult(answer="Please enter a question first.", context="")
    if store.is_empty():
        return AnswerResult(
            answer="The knowledge graph is empty. Ingest a corpus first.",
            context="",
        )

    context = store.to_text(max_chars=MAX_QA_CONTEXT_CHARS)
    try:
        answer = _QA_CHAIN.invoke(
            {"graph_context": context, "question": question}
        ).strip()
    except Exception as exc:
        raise QuestionAnsweringError(str(exc)) from exc

    return AnswerResult(answer=answer, context=context)
