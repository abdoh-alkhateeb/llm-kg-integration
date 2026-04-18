import os
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

SYSTEM_PROMPT = """You extract knowledge graph data from plain text.
Return only valid JSON with this exact shape:
{
  "entities": [
    {"id": "canonical entity name", "type": "person|organization|place|concept|event|other", "description": "short description"}
  ],
  "relationships": [
    {"source": "entity id", "target": "entity id", "relation": "short verb phrase", "evidence": "short source text quote or paraphrase"}
  ]
}
Rules:
- Use entity ids exactly as they appear in relationships.
- Include only facts supported by the corpus.
- Keep relation names concise, lowercase, and human readable.
- If there is not enough information, return empty arrays.
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=SYSTEM_PROMPT),
        ("human", "Corpus:\n{corpus}"),
    ]
)

LLM = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
    format="json",
)

EXTRACTION_CHAIN = PROMPT | LLM | JsonOutputParser()


def extract_knowledge_graph(corpus: str) -> dict[str, list[dict[str, str]]]:
    text = corpus.strip()
    if not text:
        return {"entities": [], "relationships": []}
    return _validate_extracted_graph(EXTRACTION_CHAIN.invoke({"corpus": text}))


def _validate_extracted_graph(data: dict) -> dict[str, list[dict[str, str]]]:
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

        entity_id = item.get("id", "").strip()
        if not entity_id:
            continue

        entities[entity_id] = {
            "id": entity_id,
            "type": item.get("type", "").strip() or "other",
            "description": item.get("description", "").strip(),
        }

    return list(entities.values())


def _clean_relationships(
    items: Any,
    entity_ids: set[str],
) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []

    relationships = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue

        source = item.get("source", "").strip()
        target = item.get("target", "").strip()
        relation = item.get("relation", "").strip()

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
                "evidence": item.get("evidence", "").strip(),
            }
        )

    return relationships
