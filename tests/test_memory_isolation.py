"""Memory isolation: a note written by one session must never be visible to another."""

from __future__ import annotations

from study_agent.memory.vector_store import VectorMemory


def _fallback_memory() -> VectorMemory:
    """A VectorMemory that uses the in-memory fallback (no ChromaDB required)."""
    memory = object.__new__(VectorMemory)
    memory.persist_dir = ":memory:"
    memory._client = None
    memory._collection = None
    memory._embedding_fn = None
    memory._available = False
    memory._fallback_store = []
    return memory


def test_memory_is_namespaced_per_session():
    memory = _fallback_memory()

    assert memory.store(
        "Session A note", "alpha", "what is RAG?", 0.9, "session-a", "t1"
    )
    assert memory.store(
        "Session B note", "beta", "what is RAG?", 0.9, "session-b", "t2"
    )

    assert memory.count("session-a") == 1
    assert memory.count("session-b") == 1
    assert memory.list_topics("session-a") == ["alpha"]

    hits = memory.retrieve("what is RAG?", session_id="session-a")
    assert [hit["content"] for hit in hits] == ["Session A note"]


def test_memory_requires_a_session_to_read():
    memory = _fallback_memory()
    memory.store("Session A note", "alpha", "q", 0.9, "session-a", "t1")

    assert memory.retrieve("q") == []
    assert memory.count() == 0
    assert memory.list_topics() == []


def test_low_quality_notes_are_not_stored():
    memory = _fallback_memory()

    assert memory.store("weak", "alpha", "q", 0.2, "session-a", "t1") is False
    assert memory.count("session-a") == 0
