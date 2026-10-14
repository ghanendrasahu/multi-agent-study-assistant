"""
llm_factory.py
──────────────
Factory for creating LLM instances.

Supports:
  1. Groq (free tier, OpenAI-compatible)
  2. OpenAI (GPT-4o-mini recommended - cheap and capable)
  3. Ollama local models (llama3, mistral) - completely free, no API key
  4. Anthropic Claude (claude-3-haiku - fast and cheap)

Usage:
    llm = get_llm()  # auto-detects from env vars
    llm = get_llm("openai")
    llm = get_llm("ollama", model="llama3")
"""

from __future__ import annotations

import os


def _timeout_seconds() -> float:
    """Per-request timeout for every LLM provider (default 60s)."""
    try:
        return max(5.0, float(os.getenv("LLM_TIMEOUT_SECONDS", "60")))
    except ValueError:
        return 60.0


def get_llm(provider: str | None = None, model: str | None = None):
    """
    Create and return a LangChain chat model instance.

    Auto-detection order:
        1. If USE_OLLAMA=true in env → Ollama
        2. If ANTHROPIC_API_KEY in env → Claude Haiku
        3. If OPENAI_API_KEY in env → GPT-4o-mini
        4. Raises ValueError with helpful message

    Args:
        provider: "openai", "ollama", or "anthropic" (overrides auto-detect)
        model: Specific model name to use

    Returns:
        A LangChain BaseChatModel instance
    """
    if provider is None:
        provider = _auto_detect_provider()

    if provider == "groq":
        return _get_groq(model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))
    elif provider == "ollama":
        return _get_ollama(model or "llama3")
    elif provider == "anthropic":
        return _get_anthropic(model or "claude-3-haiku-20240307")
    elif provider == "openai":
        return _get_openai(model or "gpt-4o-mini")
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            "Set GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, or USE_OLLAMA=true"
        )


def _auto_detect_provider() -> str:
    if os.getenv("USE_OLLAMA", "").lower() == "true":
        return "ollama"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    raise ValueError(
        "\n🔑 No LLM provider configured!\n\n"
        "Option 1 (OpenAI — recommended for demos):\n"
        "  export OPENAI_API_KEY='sk-...'\n\n"
        "Option 2 (Free local via Ollama):\n"
        "  Install Ollama from https://ollama.ai\n"
        "  Run: ollama pull llama3\n"
        "  Then: export USE_OLLAMA=true\n\n"
        "Option 3 (Anthropic):\n"
        "  export ANTHROPIC_API_KEY='sk-ant-...'\n"
    )


def _get_groq(model: str):
    try:
        from langchain_openai import ChatOpenAI

        print(f"[LLM] Using Groq: {model}")
        return ChatOpenAI(
            model=model,
            api_key=os.getenv("GROQ_API_KEY"),
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            temperature=0.3,
            max_tokens=2000,
            timeout=_timeout_seconds(),
            max_retries=1,
        )
    except ImportError:
        raise ImportError("Run: pip install langchain-openai")


def _get_openai(model: str):
    try:
        from langchain_openai import ChatOpenAI

        print(f"[LLM] Using OpenAI: {model}")
        return ChatOpenAI(
            model=model,
            temperature=0.3,  # Low temp for factual study notes
            max_tokens=2000,
            timeout=_timeout_seconds(),
            max_retries=1,
        )
    except ImportError:
        raise ImportError("Run: pip install langchain-openai")


def _get_anthropic(model: str):
    try:
        from langchain_anthropic import ChatAnthropic

        print(f"[LLM] Using Anthropic: {model}")
        return ChatAnthropic(
            model=model,
            temperature=0.3,
            max_tokens=2000,
            timeout=_timeout_seconds(),
            max_retries=1,
        )
    except ImportError:
        raise ImportError("Run: pip install langchain-anthropic")


def _get_ollama(model: str):
    try:
        from langchain_ollama import ChatOllama

        print(f"[LLM] Using Ollama local model: {model}")
        return ChatOllama(
            model=model,
            temperature=0.3,
            client_kwargs={"timeout": _timeout_seconds()},
        )
    except ImportError:
        raise ImportError("Run: pip install langchain-ollama")
