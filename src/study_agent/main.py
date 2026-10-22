"""
main.py
â”€â”€â”€â”€â”€â”€â”€
CLI entry point for the Multi-Agent Study Assistant.

Usage:
    python main.py
    python main.py --query "Explain transformer self-attention"
    python main.py --topic "RAG" --max-iter 2
    python main.py --list-memory
"""

from __future__ import annotations

import argparse
import sys
from dotenv import load_dotenv

load_dotenv()  # Load .env file if present


def main():
    parser = argparse.ArgumentParser(
        description="ðŸ¤– Multi-Agent Study Assistant â€” GenAI + Agentic AI Demo"
    )
    parser.add_argument("--query", "-q", type=str, help="Study query/question")
    parser.add_argument("--topic", "-t", type=str, help="Topic label for memory")
    parser.add_argument("--max-iter", "-m", type=int, default=2,
                        help="Max analyst-critic iterations (default: 2)")
    parser.add_argument("--list-memory", action="store_true",
                        help="List all topics stored in memory")
    parser.add_argument("--show-graph", action="store_true",
                        help="Show the agent graph topology")
    parser.add_argument("--provider", type=str,
                        choices=["openai", "anthropic", "ollama"],
                        help="LLM provider (auto-detects from env if not set)")
    args = parser.parse_args()

    # â”€â”€ Try rich for pretty output, fall back to print â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.markdown import Markdown
        from rich.progress import   # noqa: F401
        from rich import print as rprint
        console = Console()
        USE_RICH = True
    except ImportError:
        USE_RICH = False
        console = None

    def header(text: str):
        if USE_RICH:
            console.print(Panel(f"[bold cyan]{text}[/bold cyan]", expand=False))
        else:
            print(f"\n{'='*60}\n{text}\n{'='*60}")

    def info(text: str):
        if USE_RICH:
            console.print(f"[dim]{text}[/dim]")
        else:
            print(text)

    def success(text: str):
        if USE_RICH:
            console.print(f"[green]âœ… {text}[/green]")
        else:
            print(f"âœ… {text}")

    def show_markdown(text: str):
        if USE_RICH:
            console.print(Markdown(text))
        else:
            print(text)

    # â”€â”€ Banner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    header("ðŸ¤– Multi-Agent Study Assistant")
    info("Demonstrating: GenAI + Agentic AI + RAG + Multi-Agent Orchestration")

    # â”€â”€ Initialise components â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    from study_agent.llm_factory import get_llm
    from study_agent.memory.vector_store import VectorMemory
    from study_agent.orchestrator import build_graph

    try:
        llm = get_llm(provider=args.provider)
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    memory = VectorMemory()
    graph = build_graph(llm=llm, memory=memory)

    # â”€â”€ Show graph topology â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if args.show_graph:
        print(graph.get_graph_description())
        return

    # â”€â”€ List memory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if args.list_memory:
        topics = memory.list_topics()
        count = memory.count()
        header(f"ðŸ“š Long-Term Memory ({count} notes stored)")
        if topics:
            for t in sorted(topics):
                print(f"  â€¢ {t}")
        else:
            print("  No notes stored yet. Run a query to start building memory.")
        return

    # â”€â”€ Get query â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    query = args.query
    if not query:
        print("\n" + "â”€" * 60)
        print("ðŸ’¡ Example queries:")
        print("  â€¢ Explain transformer self-attention mechanism")
        print("  â€¢ What is RAG and how does it reduce hallucination?")
        print("  â€¢ Explain the ReAct agent pattern with an example")
        print("  â€¢ What is cosine similarity and why use it for embeddings?")
        print("  â€¢ Explain LangGraph StateGraph with nodes and edges")
        print("â”€" * 60 + "\n")
        query = input("Enter your study query: ").strip()
        if not query:
            print("No query provided. Exiting.")
            sys.exit(0)

    topic = args.topic or query[:50]

    # â”€â”€ Run the agent graph â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    header(f"ðŸ“– Studying: {query}")

    agent_steps = []

    def progress_callback(agent: str, status: str):
        agent_steps.append((agent, status))
        emoji = {
            "researcher": "ðŸ”",
            "analyst": "âœï¸",
            "critic": "ðŸ”Ž",
            "finaliser": "âœ¨"
        }.get(agent.lower(), "âš™ï¸")
        if USE_RICH:
            console.print(f"  {emoji} [bold]{agent.upper()}[/bold]: {status}")
        else:
            print(f"  {emoji} {agent.upper()}: {status}")

    try:
        result = graph.invoke(
            query=query,
            topic=topic,
            max_iterations=args.max_iter,
            progress_callback=progress_callback,
        )
    except Exception as e:
        print(f"\nâŒ Agent pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # â”€â”€ Display final answer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    header("ðŸ“ Final Study Note")
    show_markdown(result["final_answer"])

    # â”€â”€ Summary stats â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n" + "â”€" * 60)
    info(f"Quality score: {result['critique_score']:.2f}/1.00")
    info(f"Iterations: {result['iteration_count']}")
    info(f"Memory notes stored: {memory.count()}")
    info(f"Session ID: {result['session_id']}")
    success("Study note saved to long-term memory!")

    # â”€â”€ Option to query memory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n" + "â”€" * 60)
    another = input("\nðŸ“š Ask another question? (or press Enter to exit): ").strip()
    if another:
        # Recursive call for another query (demonstrates memory retrieval)
        import subprocess
        subprocess.run([sys.executable, __file__, "--query", another,
                        "--provider", args.provider or "openai"])


if __name__ == "__main__":
    main()
