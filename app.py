import gradio as gr
import networkx as nx

graph = nx.Graph()

with gr.Blocks(title="NeuroGraph") as demo:
    gr.Markdown("# NeuroGraph")
    gr.Markdown("""
    A lightweight integration of large language models and knowledge graphs.

    ### Supported capabilities
    - Corpus ingestion
    - Automated knowledge graph construction
    - Graph-enhanced LLM reasoning
    - Semantic querying
    """)

    with gr.Tabs():
        with gr.Tab("📚 Ingest Corpus"):
            corpus_input = gr.Textbox(
                label="Corpus Input",
                placeholder="Paste documents, notes, articles, transcripts, etc.",
                lines=12,
            )

            ingest_btn = gr.Button("Process Corpus")

            ingest_status = gr.Textbox(label="Pipeline Status")
            graph_output = gr.Plot(label="Knowledge Graph")

            # ingest_btn.click(
            #     fn=ingest_corpus,
            #     inputs=corpus_input,
            #     outputs=[ingest_status, graph_output],
            # )

        with gr.Tab("💬 Ask Questions"):
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

            # ask_btn.click(
            #     fn=ask_graph,
            #     inputs=question_input,
            #     outputs=answer_output,
            # )

if __name__ == "__main__":
    demo.launch()
