"""Interactive knowledge graph visualization rendered as a pyvis network.

Produces a self-contained HTML fragment (an `<iframe srcdoc="...">`)
suitable for embedding directly in a Gradio `gr.HTML` component. Using
pyvis (which wraps vis.js) gives draggable nodes, scroll-zoom, hover
tooltips, click-to-select highlighting, and a physics simulation that
lays the graph out automatically — none of which the prior matplotlib
renderer could provide.
"""

from __future__ import annotations

import html as _html
from typing import Final

import networkx as nx
from pyvis.network import Network

TYPE_COLORS: Final[dict[str, str]] = {
    "person":       "#4c78a8",
    "organization": "#54a24b",
    "place":        "#f58518",
    "concept":      "#b279a2",
    "event":        "#e45756",
    "other":        "#7f7f7f",
}

_FRAME_HEIGHT_PX: Final[int] = 620
_NETWORK_HEIGHT_PX: Final[int] = 600

_NETWORK_OPTIONS: Final[str] = """
{
  "nodes": {
    "borderWidth": 2,
    "borderWidthSelected": 4,
    "shape": "dot",
    "font": {
      "size": 15,
      "color": "#1f2937",
      "face": "Inter, system-ui, sans-serif",
      "strokeWidth": 4,
      "strokeColor": "#ffffff",
      "vadjust": 0
    },
    "shadow": {
      "enabled": true,
      "color": "rgba(15, 23, 42, 0.18)",
      "size": 8,
      "x": 1,
      "y": 2
    },
    "scaling": {
      "min": 18,
      "max": 42,
      "label": {"enabled": true, "min": 12, "max": 22}
    }
  },
  "edges": {
    "color": {
      "color": "#94a3b8",
      "highlight": "#4338ca",
      "hover": "#6366f1",
      "inherit": false
    },
    "smooth": {"enabled": true, "type": "continuous", "roundness": 0.2},
    "font": {
      "size": 12,
      "color": "#475569",
      "face": "Inter, system-ui, sans-serif",
      "strokeWidth": 4,
      "strokeColor": "#ffffff",
      "align": "middle"
    },
    "arrows": {
      "to": {"enabled": true, "scaleFactor": 0.65, "type": "arrow"}
    },
    "width": 1.5,
    "hoverWidth": 2.5,
    "selectionWidth": 2.5
  },
  "physics": {
    "enabled": true,
    "solver": "barnesHut",
    "barnesHut": {
      "gravitationalConstant": -12000,
      "centralGravity": 0.25,
      "springLength": 200,
      "springConstant": 0.04,
      "damping": 0.5,
      "avoidOverlap": 0.6
    },
    "stabilization": {
      "enabled": true,
      "iterations": 300,
      "updateInterval": 25,
      "fit": true
    },
    "minVelocity": 0.5
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 150,
    "navigationButtons": true,
    "keyboard": false,
    "multiselect": false,
    "hideEdgesOnDrag": false,
    "dragNodes": true,
    "dragView": true,
    "zoomView": true,
    "selectConnectedEdges": true
  },
  "layout": {"improvedLayout": true}
}
"""


def render_graph(graph: nx.DiGraph) -> str:
    """Render `graph` as an interactive HTML iframe ready for `gr.HTML`."""
    if graph.number_of_nodes() == 0:
        return _empty_state_html()

    net = Network(
        height=f"{_NETWORK_HEIGHT_PX}px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#1f2937",
        directed=True,
        cdn_resources="in_line",
        notebook=False,
    )
    net.set_options(_NETWORK_OPTIONS)

    for node_id, data in graph.nodes(data=True):
        entity_type = _normalize_type(data.get("type"))
        description = (data.get("description") or "").strip()
        degree = graph.in_degree(node_id) + graph.out_degree(node_id)
        color = TYPE_COLORS[entity_type]
        net.add_node(
            node_id,
            label=node_id,
            title=_tooltip_html(node_id, entity_type, description),
            color={
                "background": color,
                "border": "#1f2937",
                "highlight": {"background": color, "border": "#4338ca"},
                "hover": {"background": color, "border": "#1f2937"},
            },
            value=degree + 1,
        )

    for source, target, data in graph.edges(data=True):
        relation = (data.get("relation") or "related to").strip() or "related to"
        net.add_edge(source, target, label=relation, arrows="to")

    return _wrap_in_iframe(net.generate_html(notebook=False))


def _normalize_type(entity_type: str | None) -> str:
    key = (entity_type or "other").strip().lower()
    return key if key in TYPE_COLORS else "other"


def _tooltip_html(node_id: str, entity_type: str, description: str) -> str:
    body = description or "no description"
    return (
        '<div style="font-family:Inter,system-ui,sans-serif;'
        "max-width:280px;padding:6px 8px;line-height:1.4;\">"
        f'<div style="font-weight:600;color:#1f2937;font-size:13px;">'
        f"{_html.escape(node_id)}</div>"
        f'<div style="color:#6b7280;font-size:11px;text-transform:uppercase;'
        f'letter-spacing:0.04em;margin:2px 0 4px;">{_html.escape(entity_type)}</div>'
        f'<div style="color:#374151;font-size:12px;">{_html.escape(body)}</div>'
        "</div>"
    )


def _wrap_in_iframe(network_html: str) -> str:
    """Encode the pyvis page into an iframe so it can't bleed into Gradio's DOM."""
    safe = network_html.replace("&", "&amp;").replace('"', "&quot;")
    return (
        f'<iframe srcdoc="{safe}" '
        f'style="width:100%;height:{_FRAME_HEIGHT_PX}px;border:0;'
        f"border-radius:8px;background:#ffffff;"
        f'box-shadow:0 1px 3px rgba(15,23,42,0.06);" '
        f'sandbox="allow-scripts allow-same-origin"></iframe>'
    )


def _empty_state_html() -> str:
    return (
        '<div style="display:flex;align-items:center;justify-content:center;'
        f"height:{_FRAME_HEIGHT_PX}px;background:#ffffff;"
        "border:1px dashed #cbd5e1;border-radius:8px;"
        'color:#64748b;font-family:Inter,system-ui,sans-serif;'
        'font-size:14px;text-align:center;line-height:1.6;">'
        "<div>"
        "The knowledge graph is empty.<br>"
        "<span style=\"color:#94a3b8;\">"
        "Ingest a corpus on the LLM-augmented KG tab to populate it."
        "</span>"
        "</div></div>"
    )
