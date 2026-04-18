import os

import networkx as nx
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

SYSTEM_PROMPT = """Answer questions using only the knowledge graph context.
If the graph does not contain enough information, say that the graph does not contain enough information.
Keep the answer concise.
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=SYSTEM_PROMPT),
        ("human", "Knowledge graph:\n{graph_context}\n\nQuestion:\n{question}"),
    ]
)

LLM = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)

QA_CHAIN = PROMPT | LLM | StrOutputParser()


def answer_question(graph: nx.DiGraph, question: str) -> str:
    question = question.strip()
    if not question:
        return "Enter a question first."
    if graph.number_of_nodes() == 0:
        return "The knowledge graph is empty. Ingest corpus text first."

    return QA_CHAIN.invoke(
        {
            "graph_context": _graph_to_text(graph),
            "question": question,
        }
    ).strip()


def _graph_to_text(graph: nx.DiGraph) -> str:
    lines = ["Entities:"]
    for node_id, data in graph.nodes(data=True):
        entity_type = data.get("type", "other")
        description = data.get("description", "")
        lines.append(f"- {node_id} ({entity_type}): {description}")

    lines.append("\nRelationships:")
    for source, target, data in graph.edges(data=True):
        relation = data.get("relation", "related to")
        evidence = data.get("evidence", "")
        lines.append(f"- {source} --{relation}--> {target}. Evidence: {evidence}")

    return "\n".join(lines)
