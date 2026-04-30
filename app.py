import gradio as gr
import matplotlib.pyplot as plt
import networkx as nx

from kg_extractor import extract_knowledge_graph
from kg_qa import answer_question

graph = nx.DiGraph()


def ingest_corpus(corpus: str):
    try:
        extracted_graph = extract_knowledge_graph(corpus)
    except Exception as exc:
        return f"Extraction failed: {exc}", draw_graph()

    for entity in extracted_graph["entities"]:
        graph.add_node(
            entity["id"],
            type=entity["type"],
            description=entity["description"],
        )

    for relationship in extracted_graph["relationships"]:
        graph.add_edge(
            relationship["source"],
            relationship["target"],
            relation=relationship["relation"],
            evidence=relationship["evidence"],
        )

    entity_count = len(extracted_graph["entities"])
    relationship_count = len(extracted_graph["relationships"])
    return (
        "LLM-augmented KG complete: "
        f"added {entity_count} entities and {relationship_count} relationships.",
        draw_graph(),
    )


def draw_graph():
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_axis_off()

    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No graph data yet.", ha="center", va="center")
        return fig

    pos = nx.spring_layout(graph, seed=7)
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=1800,
        node_color="#d9ead3",
        edgecolors="#38761d",
    )
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=9)
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=16,
        edge_color="#666666",
    )
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        ax=ax,
        edge_labels=nx.get_edge_attributes(graph, "relation"),
        font_size=8,
    )

    fig.tight_layout()
    return fig


def ask_graph(question: str):
    try:
        return answer_question(graph, question)
    except Exception as exc:
        return f"Question answering failed: {exc}"


with gr.Blocks(title="NeuroGraph") as demo:
    gr.Markdown("# NeuroGraph")
    gr.Markdown("""
    A lightweight integration of large language models and knowledge graphs.

    ### Applications
    - LLM-augmented KG: corpus ingestion and automated knowledge graph construction
    - KG-enhanced LLM: graph-grounded question answering over extracted entities and relationships
    """)

    with gr.Tabs():
        with gr.Tab("LLM-augmented KG: Ingest Corpus"):
            corpus_input = gr.Textbox(
                label="Corpus Input",
                placeholder="Paste documents, notes, articles, transcripts, etc.",
                lines=12,
            )

            ingest_btn = gr.Button("Process Corpus")

            ingest_status = gr.Textbox(label="Pipeline Status")
            graph_output = gr.Plot(label="Knowledge Graph")

            ingest_btn.click(
                fn=ingest_corpus,
                inputs=corpus_input,
                outputs=[ingest_status, graph_output],
            )

        with gr.Tab("KG-enhanced LLM: Ask Questions"):
            question_input = gr.Textbox(
                label="Question",
                placeholder="Ask something about the knowledge graph...",
                lines=3,
            )

            ask_btn = gr.Button("Ask")

            answer_output = gr.Textbox(
                label="LLM Response",
                lines=10,
            )

            ask_btn.click(
                fn=ask_graph,
                inputs=question_input,
                outputs=answer_output,
            )

if __name__ == "__main__":
    demo.launch()
