# NeuroGraph

A Gradio workbench for the two-way integration of large language models and knowledge graphs:

- **LLM-augmented KG** — use a language model to extract a structured knowledge graph from text.
- **KG-enhanced LLM** — answer questions grounded in the extracted graph.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate           # Windows (PowerShell)
# source .venv/bin/activate      # macOS / Linux

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Pull the default LLM
ollama pull qwen2.5:3b
```

## Run

```bash
python app.py
```

The UI opens at <http://localhost:7860>.

## Configuration

Override defaults with environment variables:

| Variable               | Default                  | Purpose                             |
| ---------------------- | ------------------------ | ----------------------------------- |
| `OLLAMA_BASE_URL`      | `http://localhost:11434` | Ollama API endpoint                 |
| `OLLAMA_MODEL`         | `qwen2.5:3b`             | Model used for extraction and QA    |
| `MAX_CORPUS_CHARS`     | `20000`                  | Max characters processed per ingest |
| `MAX_QA_CONTEXT_CHARS` | `16000`                  | Max graph context sent per question |
