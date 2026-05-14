"""NeuroGraph: a Gradio workbench for LLM-KG integration.

Hosts two applications side-by-side over a shared, per-session knowledge graph:

    * LLM-augmented KG  — extract a structured graph from text using an LLM.
    * KG-enhanced LLM   — answer questions grounded in the extracted graph.
"""

from __future__ import annotations

import html

import gradio as gr
import networkx as nx

from config import MAX_CORPUS_CHARS, OLLAMA_MODEL
from kg_enhanced_llm import (
    AnswerResult,
    QuestionAnsweringError,
    answer_question,
)
from kg_store import KnowledgeGraphStore
from kg_visualizer import render_graph
from llm_augmented_kg import ExtractionError, extract_knowledge_graph

PLACEHOLDER_SAMPLE = "— Choose a sample —"

SAMPLE_CORPORA: dict[str, str] = {
    PLACEHOLDER_SAMPLE: "",
    "Albert Einstein (biography)": (
        "Albert Einstein was a German-born theoretical physicist who developed "
        "the theory of relativity. He was born in Ulm, Germany in 1879 and later "
        "emigrated to the United States, where he worked at the Institute for "
        "Advanced Study in Princeton. Einstein received the Nobel Prize in "
        "Physics in 1921 for his discovery of the photoelectric effect."
    ),
    "Apollo 11 (history)": (
        "Apollo 11 was a NASA spaceflight that landed the first humans on the "
        "Moon on July 20, 1969. Commander Neil Armstrong and lunar module pilot "
        "Buzz Aldrin walked on the lunar surface while Michael Collins piloted "
        "the Command Module Columbia in lunar orbit. The mission launched from "
        "Kennedy Space Center in Florida."
    ),
    "The World Wide Web (technology)": (
        "In 1989, Tim Berners-Lee proposed the World Wide Web while working at "
        "CERN, a research organization in Switzerland. The first website went "
        "live in 1991. In 1994, Berners-Lee founded the World Wide Web "
        "Consortium at MIT to develop open standards for the web."
    ),
}

EXAMPLE_QUESTIONS: list[list[str]] = [
    ["What organizations are mentioned in the graph?"],
    ["Who is the most connected entity in the graph?"],
    ["Summarize the relationships between people and organizations."],
]

ENTITY_HEADERS = ["Entity", "Type", "Description"]
RELATIONSHIP_HEADERS = ["Source", "Relation", "Target"]

INITIAL_STATUS = "Ready. Paste or select a corpus to begin."


def _stats_markdown(store: KnowledgeGraphStore) -> str:
    return (
        f"**Entities** {store.entity_count} "
        f"&nbsp;·&nbsp; "
        f"**Relationships** {store.relationship_count} "
        f"&nbsp;·&nbsp; "
        f"**Model** `{OLLAMA_MODEL}`"
    )


def _status(message: str, level: str = "info") -> str:
    palette = {
        "info":    ("#1f2937", "#f3f4f6"),
        "success": ("#0f5132", "#d1e7dd"),
        "warn":    ("#664d03", "#fff3cd"),
        "error":   ("#842029", "#f8d7da"),
    }
    fg, bg = palette.get(level, palette["info"])
    return (
        f'<div style="padding:0.65rem 0.9rem;border-radius:8px;'
        f"background:{bg};color:{fg};font-size:0.92rem;"
        f'border:1px solid rgba(0,0,0,0.05);">{html.escape(message)}</div>'
    )


def _ingest(corpus: str, store: KnowledgeGraphStore):
    text = corpus.strip()
    if not text:
        return _ingest_outputs(
            store, _status("Enter or select a corpus before processing.", "warn")
        )

    truncation_note = ""
    if len(text) > MAX_CORPUS_CHARS:
        truncation_note = (
            f" Corpus exceeded {MAX_CORPUS_CHARS:,} characters and was truncated."
        )

    try:
        extracted = extract_knowledge_graph(text)
    except ExtractionError as exc:
        return _ingest_outputs(store, _status(f"Extraction failed: {exc}", "error"))

    entities = extracted["entities"]
    relationships = extracted["relationships"]
    if not entities:
        return _ingest_outputs(
            store,
            _status(
                "No entities were extracted. "
                "Try a longer or more concrete passage." + truncation_note,
                "warn",
            ),
        )

    result = store.ingest(entities, relationships)
    summary = (
        f"Extracted {len(entities)} entities and {len(relationships)} relationships. "
        f"Added {result.entities_added} new entities and "
        f"{result.relationships_added} new relationships." + truncation_note
    )
    return _ingest_outputs(store, _status(summary, "success"))


def _ingest_outputs(store: KnowledgeGraphStore, status_html: str):
    return (
        store,
        status_html,
        render_graph(store.graph),
        store.entities_table(),
        store.relationships_table(),
        _stats_markdown(store),
    )


def _clear(store: KnowledgeGraphStore):
    store.clear()
    return _ingest_outputs(store, _status("Knowledge graph cleared.", "info"))


def _load_sample(name: str):
    """Populate the corpus textarea from the sample dropdown.

    Returning `gr.update()` (no change) for the placeholder option prevents
    re-selecting "— Choose a sample —" from wiping a corpus the user is
    actively editing.
    """
    if not name or name == PLACEHOLDER_SAMPLE:
        return gr.update()
    return SAMPLE_CORPORA.get(name, gr.update())


def _ask(question: str, store: KnowledgeGraphStore):
    try:
        result: AnswerResult = answer_question(store, question)
    except QuestionAnsweringError as exc:
        return f"Question answering failed: {exc}", ""
    return result.answer, result.context


THEME = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
    font=[
        gr.themes.GoogleFont("Inter"),
        "system-ui",
        "-apple-system",
        "Segoe UI",
        "Roboto",
        "Helvetica Neue",
        "sans-serif",
    ],
    font_mono=[
        gr.themes.GoogleFont("Source Code Pro"),
        "ui-monospace",
        "SFMono-Regular",
        "Consolas",
        "Menlo",
        "monospace",
    ],
)

CSS = """
:root, .gradio-container {
    font-feature-settings: "liga" 0, "calt" 0, "ss01" 0, "ss02" 0,
                            "ss03" 0, "ss04" 0, "ss05" 0, "zero" 0;
}
.gradio-container code,
.gradio-container pre,
.gradio-container kbd,
.gradio-container samp {
    font-feature-settings: "liga" 0, "calt" 0, "ss01" 0, "ss02" 0,
                            "ss03" 0, "ss04" 0, "ss05" 0, "zero" 0;
    font-variant-ligatures: none;
}

.neurograph-title h1 {
    margin: 0 0 0.15rem 0;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.neurograph-title p { margin: 0 0 0.25rem 0; }

.stats-bar {
    padding: 0.6rem 0.9rem;
    border-radius: 8px;
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    color: var(--body-text-color);
}
.stats-bar p { color: var(--body-text-color); margin: 0; }
.stats-bar code {
    color: var(--body-text-color);
    background: var(--background-fill-primary);
    padding: 0.05rem 0.4rem;
    border-radius: 4px;
    border: 1px solid var(--border-color-primary);
    font-size: 0.88em;
}

.tab-intro {
    color: var(--body-text-color-subdued);
    margin-top: 0.25rem;
    margin-bottom: 0.5rem;
}

footer { visibility: hidden; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="NeuroGraph") as demo:
        store_state = gr.State(KnowledgeGraphStore())

        gr.Markdown(
            "# NeuroGraph\n"
            "A workbench for the two-way integration of large language models "
            "and knowledge graphs.",
            elem_classes=["neurograph-title"],
        )
        stats = gr.Markdown(
            _stats_markdown(KnowledgeGraphStore()),
            elem_classes=["stats-bar"],
        )

        graph_plot = gr.Plot(
            value=render_graph(nx.DiGraph()),
            label="Knowledge Graph",
            show_label=False,
        )

        with gr.Tabs():
            with gr.Tab("LLM-augmented KG · Build"):
                gr.Markdown(
                    "Use a language model to extract entities and relationships "
                    "from a corpus, then merge them into the knowledge graph.",
                    elem_classes=["tab-intro"],
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        sample_dropdown = gr.Dropdown(
                            label="Sample corpus",
                            choices=list(SAMPLE_CORPORA.keys()),
                            value=PLACEHOLDER_SAMPLE,
                            interactive=True,
                        )
                        corpus_input = gr.Textbox(
                            label="Corpus",
                            placeholder=(
                                "Paste documents, notes, articles, transcripts…"
                            ),
                            lines=12,
                        )
                    with gr.Column(scale=1):
                        ingest_btn = gr.Button("Process Corpus", variant="primary")
                        clear_btn = gr.Button("Clear Graph", variant="secondary")
                        ingest_status = gr.Markdown(_status(INITIAL_STATUS, "info"))

                with gr.Row():
                    entities_df = gr.Dataframe(
                        headers=ENTITY_HEADERS,
                        datatype=["str", "str", "str"],
                        value=[],
                        label="Entities",
                        interactive=False,
                        wrap=True,
                    )
                    relationships_df = gr.Dataframe(
                        headers=RELATIONSHIP_HEADERS,
                        datatype=["str", "str", "str"],
                        value=[],
                        label="Relationships",
                        interactive=False,
                        wrap=True,
                    )

            with gr.Tab("KG-enhanced LLM · Ask"):
                gr.Markdown(
                    "Ask questions whose answers are grounded only in facts "
                    "present in the current knowledge graph.",
                    elem_classes=["tab-intro"],
                )
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="Ask a question about the entities in the graph…",
                    lines=3,
                )
                ask_btn = gr.Button("Ask", variant="primary")
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=6,
                    interactive=False,
                )
                with gr.Accordion(
                    "Show graph context sent to the model", open=False
                ):
                    context_output = gr.Textbox(
                        label="Context",
                        lines=15,
                        interactive=False,
                    )
                gr.Examples(
                    examples=EXAMPLE_QUESTIONS,
                    inputs=question_input,
                    label="Example questions",
                )

        sample_dropdown.change(
            fn=_load_sample,
            inputs=sample_dropdown,
            outputs=corpus_input,
        )
        ingest_btn.click(
            fn=_ingest,
            inputs=[corpus_input, store_state],
            outputs=[
                store_state,
                ingest_status,
                graph_plot,
                entities_df,
                relationships_df,
                stats,
            ],
        )
        clear_btn.click(
            fn=_clear,
            inputs=store_state,
            outputs=[
                store_state,
                ingest_status,
                graph_plot,
                entities_df,
                relationships_df,
                stats,
            ],
        )
        ask_btn.click(
            fn=_ask,
            inputs=[question_input, store_state],
            outputs=[answer_output, context_output],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(theme=THEME, css=CSS)
