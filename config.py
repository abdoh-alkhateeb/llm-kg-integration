"""Shared configuration for the LLM-KG integration apps."""

from __future__ import annotations

import os

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

MAX_CORPUS_CHARS: int = int(os.getenv("MAX_CORPUS_CHARS", "20000"))
MAX_QA_CONTEXT_CHARS: int = int(os.getenv("MAX_QA_CONTEXT_CHARS", "16000"))
