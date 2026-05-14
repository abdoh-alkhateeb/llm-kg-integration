"""Knowledge graph storage backed by networkx.

Wraps a `networkx.DiGraph` to provide a small, intention-revealing API used
by both applications (LLM-augmented KG and KG-enhanced LLM). Keeping graph
mutations behind this surface makes the rest of the codebase easier to test
and lets us evolve the storage layer without rewriting callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx

ENTITY_TYPES: tuple[str, ...] = (
    "person",
    "organization",
    "place",
    "concept",
    "event",
    "other",
)


@dataclass(frozen=True)
class IngestionResult:
    """Summary of what an ingestion call changed in the store."""

    entities_added: int
    entities_updated: int
    relationships_added: int
    relationships_updated: int


class KnowledgeGraphStore:
    """In-memory knowledge graph store with a typed, intention-revealing API."""

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    @property
    def entity_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def relationship_count(self) -> int:
        return self._graph.number_of_edges()

    def is_empty(self) -> bool:
        return self.entity_count == 0

    def clear(self) -> None:
        self._graph.clear()

    def ingest(
        self,
        entities: Iterable[dict[str, str]],
        relationships: Iterable[dict[str, str]],
    ) -> IngestionResult:
        """Merge entities and relationships into the graph and return a diff summary."""
        entities_added, entities_updated = self._add_entities(entities)
        relationships_added, relationships_updated = self._add_relationships(relationships)
        return IngestionResult(
            entities_added=entities_added,
            entities_updated=entities_updated,
            relationships_added=relationships_added,
            relationships_updated=relationships_updated,
        )

    def _add_entities(self, entities: Iterable[dict[str, str]]) -> tuple[int, int]:
        added = 0
        updated = 0
        for entity in entities:
            entity_id = entity["id"]
            if self._graph.has_node(entity_id):
                updated += 1
            else:
                added += 1
            self._graph.add_node(
                entity_id,
                type=entity.get("type", "other") or "other",
                description=entity.get("description", ""),
            )
        return added, updated

    def _add_relationships(
        self, relationships: Iterable[dict[str, str]]
    ) -> tuple[int, int]:
        added = 0
        updated = 0
        for rel in relationships:
            source = rel["source"]
            target = rel["target"]
            if not self._graph.has_node(source) or not self._graph.has_node(target):
                continue
            if self._graph.has_edge(source, target):
                updated += 1
            else:
                added += 1
            self._graph.add_edge(
                source,
                target,
                relation=rel.get("relation", "related to"),
            )
        return added, updated

    def entities_table(self) -> list[list[str]]:
        """Return entities as `[id, type, description]` rows for display."""
        return [
            [node_id, data.get("type", ""), data.get("description", "")]
            for node_id, data in self._graph.nodes(data=True)
        ]

    def relationships_table(self) -> list[list[str]]:
        """Return relationships as `[source, relation, target]` rows."""
        return [
            [src, data.get("relation", ""), tgt]
            for src, tgt, data in self._graph.edges(data=True)
        ]

    def to_text(self, max_chars: int | None = None) -> str:
        """Serialize the graph to a plain-text block suitable for LLM context.

        When `max_chars` is set and exceeded, truncation happens at the
        nearest preceding line boundary so the model never sees a half-fact.
        """
        if self.is_empty():
            return "The knowledge graph is empty."

        lines = ["Entities:"]
        for node_id, data in self._graph.nodes(data=True):
            entity_type = data.get("type", "other") or "other"
            description = data.get("description", "") or "no description"
            lines.append(f"- {node_id} ({entity_type}): {description}")

        lines.append("")
        lines.append("Relationships:")
        for source, target, data in self._graph.edges(data=True):
            relation = data.get("relation", "related to") or "related to"
            lines.append(f"- {source} --{relation}--> {target}")

        text = "\n".join(lines)
        if max_chars is not None and len(text) > max_chars:
            head = text[:max_chars]
            cutoff = head.rfind("\n")
            if cutoff > 0:
                head = head[:cutoff]
            text = head.rstrip() + "\n... [graph truncated]"
        return text
