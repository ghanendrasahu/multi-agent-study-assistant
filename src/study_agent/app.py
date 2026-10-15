"""Streamlit web UI for the Multi-Agent Study Assistant."""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

try:
    from study_agent import ui
except ImportError:  # direct `streamlit run src/study_agent/app.py`
    import ui

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
ui.setup("Study Assistant", icon=ui.app_icon(APP_DIR.parent.parent / "assets"))

st.markdown(
    """
    <style>
      .agent-row { display: flex; align-items: center; gap: .6rem; padding: .55rem .8rem;
                   border: 1px solid #232B3D; border-radius: 10px; margin-bottom: .4rem;
                   background: #111521; }
      .agent-row.done { border-color: rgba(70, 211, 154, .35); }
      .agent-row.active { border-color: rgba(139, 147, 255, .5); }
      .agent-name { font-size: 13px; font-weight: 600; color: #E8EBF5; min-width: 92px; }
      .agent-state { font-size: 13px; color: #9AA4B8; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "result" not in st.session_state:
    st.session_state.result = None
if "history" not in st.session_state:
    st.session_state.history = []
# Memory is namespaced per browser session: one visitor's notes are never
# retrieved into another visitor's prompts.
if "memory_session_id" not in st.session_state:
    st.session_state.memory_session_id = uuid.uuid4().hex


@st.cache_resource(show_spinner=False)
def _get_memory():
    """One Chroma client and embedding model per process."""
    from study_agent.memory.vector_store import VectorMemory

    return VectorMemory()


with st.sidebar:
    ui.sidebar_brand("Study Assistant", "Multi-agent research")
    provider = st.selectbox(
        "Provider",
        ["groq", "openai", "anthropic", "ollama"],
        help="Groq is hosted and free to start. Ollama runs locally.",
    )
    model_map = {
        "groq": ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"],
        "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"],
        "ollama": ["llama3", "mistral", "phi3", "gemma2"],
    }
    model = st.selectbox("Model", model_map[provider])
    max_iter = st.slider("Revision rounds", 1, 3, 2)
    quality_threshold = st.slider("Quality threshold", 0.5, 0.95, 0.75, 0.05)

    with st.expander("How the agents work"):
        st.markdown(
            "**Researcher** gathers sources and stored notes.\n\n"
            "**Analyst** writes a structured study note.\n\n"
            "**Critic** scores the draft and sends it back if it falls short.\n\n"
            "**Finaliser** polishes the note and saves it to memory."
        )
        st.code(
            "researcher -> analyst -> critic\n"
            "                 ^        |\n"
            "                 +--------+  score below threshold\n"
            "                          |\n"
            "                     finaliser",
            language=None,
        )

    with st.expander("What this demo shows"):
        st.markdown(
            "- Retrieval augmented generation\n"
            "- Reflexion and critic loops\n"
            "- Long-term memory with ChromaDB\n"
            "- Conditional routing between agents"
        )

ui.page_header(
    "Study assistant",
    "Ask a technical question and watch four agents research, draft, review and refine the answer.",
    eyebrow="MULTI-AGENT RESEARCH",
    badge="4 agents ready",
    icon_name="bot",
)

ui.section(
    "Start with a question",
    "Pick a prompt or write your own. The answer is saved to long-term memory when the run finishes.",
)

EXAMPLES = [
    "Explain transformer self-attention",
    "What is RAG and how does it reduce hallucination?",
    "Explain the ReAct agent pattern",
    "What is cosine similarity and why is it used for embeddings?",
    "How does LangGraph StateGraph work?",
    "Chain-of-Thought versus Plan-and-Execute",
]

cols = st.columns(2)
for index, example in enumerate(EXAMPLES):
    if cols[index % 2].button(example, key=f"ex_{index}", width="stretch"):
        st.session_state.query_text = example

with st.form("query_form"):
    query = st.text_area(
        "Your question",
        key="query_text",
        placeholder="e.g. Explain how HNSW indexing works in vector databases",
        height=90,
    )
    topic = st.text_input(
        "Topic label (optional)",
        placeholder="e.g. Vector databases",
    )
    submitted = st.form_submit_button("Run agents", type="primary", width="stretch")

if submitted and not query.strip():
    ui.error_state(
        "Enter a question before running the agents.",
        title="Nothing to research",
        hint="Try one of the prompts above.",
    )

if submitted and query.strip():
    ui.section("Run", "The pipeline updates each agent as it finishes.")
    try:
        from study_agent.llm_factory import get_llm
        from study_agent.orchestrator import build_graph
    except Exception:
        ui.error_state(
            "The agent dependencies are not available in this environment.",
            title="Study assistant unavailable",
            hint="Install the project requirements and reload the app.",
        )
    else:
        os.environ["__STUDY_AGENT_PROVIDER"] = provider
        import study_agent.llm_factory as _lf

        original_detect = _lf._auto_detect_provider
        _lf._auto_detect_provider = lambda: provider
        try:
            llm = get_llm(provider=provider, model=model)
        except Exception:
            ui.error_state(
                "The language model could not start. Check the provider key and try again.",
                title="Model unavailable",
                hint="Groq is the quickest provider to configure.",
            )
            llm = None
        finally:
            _lf._auto_detect_provider = original_detect

        if llm is not None:
            memory = _get_memory()
            graph = build_graph(llm=llm, memory=memory)

            steps = ["researcher", "analyst", "critic", "finaliser"]
            descriptions = {
                "researcher": "Searching sources and stored notes",
                "analyst": "Writing the structured note",
                "critic": "Scoring and reviewing the draft",
                "finaliser": "Polishing and saving to memory",
            }
            placeholders = {}
            for agent in steps:
                placeholders[agent] = st.empty()
                placeholders[agent].markdown(
                    f'<div class="agent-row"><span class="agent-name">{agent.title()}</span>'
                    f'<span class="agent-state">{descriptions[agent]}</span></div>',
                    unsafe_allow_html=True,
                )

            def progress_callback(agent: str, status: str) -> None:
                key = agent.lower()
                if key in placeholders:
                    placeholders[key].markdown(
                        f'<div class="agent-row done"><span class="agent-name">'
                        f'{key.title()}</span><span class="agent-state">{status}</span></div>',
                        unsafe_allow_html=True,
                    )

            with st.spinner("Running the agent pipeline..."):
                try:
                    result = graph.invoke(
                        query=query.strip(),
                        topic=topic.strip() or query.strip()[:50],
                        max_iterations=max_iter,
                        session_id=st.session_state.memory_session_id,
                        progress_callback=progress_callback,
                    )
                except Exception:
                    ui.error_state(
                        "The agents stopped before finishing. Check the provider key, "
                        "then run the question again.",
                        title="Pipeline failed",
                        hint="Stored notes are unchanged.",
                    )
                    result = None

            if result is not None:
                st.session_state.result = result
                st.session_state.history.append(
                    {
                        "query": query,
                        "result": result,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    }
                )

                score = float(result.get("critique_score", 0))
                ui.section(
                    "Study note",
                    "The final answer, with the working kept in the expanders.",
                )
                ui.metric_grid(
                    [
                        {
                            "label": "Quality score",
                            "value": f"{score:.2f}",
                            "caption": "out of 1.00",
                        },
                        {
                            "label": "Revision rounds",
                            "value": str(result.get("iteration_count", 1)),
                        },
                        {
                            "label": "Stored notes",
                            "value": str(
                                memory.count(st.session_state.memory_session_id)
                            ),
                        },
                    ]
                )
                st.markdown(result["final_answer"])

                with st.expander("Researcher sources"):
                    docs = result.get("retrieved_docs", [])
                    st.markdown(docs[0] if docs else "No sources retrieved.")
                    st.markdown("**Memory context**")
                    st.markdown(
                        result.get("memory_context", "No past notes retrieved.")
                        or "No past notes retrieved."
                    )
                with st.expander("Analyst draft"):
                    st.markdown(result.get("draft_answer", "No draft stored."))
                with st.expander("Critic review"):
                    st.markdown(result.get("critique", "No review stored."))

                st.download_button(
                    "Download study note",
                    data=result["final_answer"],
                    file_name=f"study_note_{topic or 'query'}.md",
                    mime="text/markdown",
                    width="stretch",
                )

ui.section(
    "Long-term memory",
    "Notes from this visit are stored for this session and reused on later runs.",
)

try:
    mem = _get_memory()
    count = mem.count(st.session_state.memory_session_id)
    topics = mem.list_topics(st.session_state.memory_session_id)
except Exception:
    ui.empty_state(
        "Memory is unavailable",
        "The vector store could not start in this environment. Runs still work, "
        "but notes will not be saved between sessions.",
        icon_name="database",
    )
else:
    if count == 0:
        ui.empty_state(
            "No saved notes yet",
            "Run a question and the finished note will appear here for reuse.",
            icon_name="database",
        )
    else:
        st.caption(f"{count} notes stored across {len(topics)} topics.")
        left, right = st.columns([1, 2])
        with left:
            st.markdown("**Topics**")
            for name in sorted(topics):
                st.markdown(f"- {name}")
        with right:
            search_mem = st.text_input("Search memory", key="mem_search")
            if search_mem:
                results = mem.retrieve(
                    search_mem,
                    n_results=3,
                    session_id=st.session_state.memory_session_id,
                )
                if results:
                    for item in results:
                        label = f"{item['topic']} - similarity {item.get('similarity', 0):.2f}"
                        with st.expander(label):
                            st.markdown(item["content"][:500] + "...")
                else:
                    ui.empty_state(
                        "No matching notes",
                        "Try a broader phrase or a different topic.",
                        icon_name="search",
                    )

st.caption(
    "Architecture: ReAct, chain-of-thought, Reflexion, RAG, ChromaDB and "
    "LangGraph-style routing across four agents."
)
