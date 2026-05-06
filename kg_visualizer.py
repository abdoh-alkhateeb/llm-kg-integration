"""Knowledge graph visualization rendered as a matplotlib Figure.

Kept separate from the Gradio layer so that rendering can be reused, tested,
or swapped for a different backend without touching the UI.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

TYPE_COLORS: dict[str, str] = {
    "person": "#4c78a8",
    "organization": "#54a24b",
    "place": "#f58518",
    "concept": "#b279a2",
    "event": "#e45756",
    "other": "#7f7f7f",
}

EMPTY_STATE_MESSAGE = (
    "The knowledge graph is empty.\n"
    "Ingest a corpus on the LLM-augmented KG tab to populate it."
)


def render_graph(graph: nx.DiGraph) -> Figure:
    """Render `graph` as a clean, legible matplotlib figure."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_axis_off()

    if graph.number_of_nodes() == 0:
        ax.text(
            0.5,
            0.5,
            EMPTY_STATE_MESSAGE,
            ha="center",
            va="center",
            fontsize=13,
            color="#888888",
        )
        fig.tight_layout()
        return fig

    pos = _layout(graph)
    node_colors = [_color_for(data.get("type", "other")) for _, data in graph.nodes(data=True)]

    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=1700,
        node_color=node_colors,
        edgecolors="#222222",
        linewidths=1.0,
        alpha=0.95,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        ax=ax,
        labels={node: _truncate(node, 22) for node in graph.nodes()},
        font_size=9,
        font_color="#ffffff",
        font_weight="bold",
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=14,
        edge_color="#666666",
        width=1.2,
        connectionstyle="arc3,rad=0.08",
        node_size=1700,
    )
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        ax=ax,
        edge_labels={
            (s, t): _truncate(data.get("relation", ""), 24)
            for s, t, data in graph.edges(data=True)
        },
        font_size=8,
        font_color="#333333",
        label_pos=0.5,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
    )

    _draw_legend(ax, graph)
    fig.tight_layout()
    return fig


def _layout(graph: nx.DiGraph) -> dict:
    n = graph.number_of_nodes()
    return nx.spring_layout(
        graph,
        seed=42,
        k=1.6 / max(1.0, n**0.5),
        iterations=120,
    )


def _color_for(entity_type: str | None) -> str:
    key = (entity_type or "other").strip().lower()
    return TYPE_COLORS.get(key, TYPE_COLORS["other"])


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _draw_legend(ax, graph: nx.DiGraph) -> None:
    types_present = sorted(
        {(data.get("type") or "other").lower() for _, data in graph.nodes(data=True)}
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_color_for(t),
            markeredgecolor="#222222",
            markersize=10,
            label=t,
        )
        for t in types_present
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        frameon=False,
        fontsize=9,
        title="Entity types",
        title_fontsize=9,
    )
