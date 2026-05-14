"""LLM-augmented Knowledge Graph: extract a structured KG from unstructured text.

This module implements the *LLM-augmented KG* application — using a language
model to read natural-language text and produce a normalized graph of
entities and relationships ready for ingestion into [`KnowledgeGraphStore`].
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import MAX_CORPUS_CHARS, OLLAMA_BASE_URL, OLLAMA_MODEL
from kg_store import ENTITY_TYPES

SYSTEM_PROMPT = """You extract knowledge graph data from plain text.
Return only valid JSON with this exact shape:
{
  "entities": [
    {"id": "canonical entity name", "type": "person|organization|place|concept|event|other", "description": "short description"}
  ],
  "relationships": [
    {"source": "entity id", "target": "entity id", "relation": "short verb phrase"}
  ]
}
Rules:
- Use entity ids exactly as they appear in relationships.
- Include only facts supported by the corpus.
- Keep relation names concise, lowercase, and human readable.
- If there is not enough information, return empty arrays.
"""

_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=SYSTEM_PROMPT),
        ("human", "Corpus:\n{corpus}"),
    ]
)

_LLM = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    format="json",
)

_EXTRACTION_CHAIN = _PROMPT | _LLM | JsonOutputParser()


class ExtractionError(RuntimeError):
    """Raised when the underlying LLM call fails or returns unusable data."""


def extract_knowledge_graph(corpus: str) -> dict[str, list[dict[str, str]]]:
    """Extract entities and relationships from `corpus`.

    Returns a dict with `entities` and `relationships` lists, both of which
    may be empty. Raises `ExtractionError` if the LLM call itself fails.
    """
    text = corpus.strip()
    if not text:
        return {"entities": [], "relationships": []}

    if len(text) > MAX_CORPUS_CHARS:
        text = text[:MAX_CORPUS_CHARS]

    try:
        raw = _EXTRACTION_CHAIN.invoke({"corpus": text})
    except Exception as exc:
        raise ExtractionError(str(exc)) from exc

    return _validate(raw)


def _validate(data: Any) -> dict[str, list[dict[str, str]]]:
    if not isinstance(data, dict):
        return {"entities": [], "relationships": []}
    entities = _clean_entities(data.get("entities", []))
    entity_ids = {entity["id"] for entity in entities}
    relationships = _clean_relationships(data.get("relationships", []), entity_ids)
    return {"entities": entities, "relationships": relationships}


def _clean_entities(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []

    entities: dict[str, dict[str, str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue

        entity_id = str(item.get("id", "")).strip()
        if not entity_id:
            continue

        entity_type = str(item.get("type", "")).strip().lower() or "other"
        if entity_type not in ENTITY_TYPES:
            entity_type = "other"

        entities[entity_id] = {
            "id": entity_id,
            "type": entity_type,
            "description": str(item.get("description", "")).strip(),
        }

    return list(entities.values())


def _clean_relationships(
    items: Any, entity_ids: set[str]
) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []

    relationships: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue

        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        relation = str(item.get("relation", "")).strip().lower()

        if not source or not target or not relation:
            continue
        if source not in entity_ids or target not in entity_ids:
            continue

        key = (source, target, relation)
        if key in seen:
            continue
        seen.add(key)

        relationships.append(
            {
                "source": source,
                "target": target,
                "relation": relation,
            }
        )

    return relationships
