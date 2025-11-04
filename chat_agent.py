"""
chat_agent.py
-------------
LangGraph-based chat agent that uses a local Ollama model
(connected via REST API) as the LLM backend.

Features:
- Loads model + API settings from .env file
- Maintains conversation context (efficient memory)
- Integrates with LangGraph StateGraph
"""

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from ollama_client import OllamaClient

# ---------------------------------------------------------------------------
# Define the shared state passed between graph nodes
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    """
    The ChatState keeps track of:
    - messages: list of user + model messages (alternating)
    - context: Ollama token context for continuity between turns
    """
    messages: List[str]
    context: Optional[List[int]]


# ---------------------------------------------------------------------------
# Initialize the Ollama client (reads from .env automatically)
# ---------------------------------------------------------------------------
ollama = OllamaClient()


# ---------------------------------------------------------------------------
# Define the main node: handles a single prompt/response round
# ---------------------------------------------------------------------------
def ollama_node(state: ChatState) -> ChatState:
    """
    LangGraph node that:
    1. Reads the latest user message from state["messages"][-1]
    2. Sends it to the local Ollama model
    3. Appends the model's reply
    4. Updates the context for continuity
    """
    user_prompt = state["messages"][-1]
    previous_context = state.get("context")

    # Call local Ollama model
    result = ollama.generate(
        prompt=user_prompt,
        context=previous_context,
        stream=False  # for simplicity, non-streaming here
    )

    # Extract relevant info
    reply = result.get("response", "")
    new_context = result.get("context", previous_context)

    # Append model reply and update context
    state["messages"].append(reply)
    state["context"] = new_context

    # Optional: Debug / metrics output
    if "eval_count" in result and "total_duration" in result:
        print(f"[Ollama] Generated {result['eval_count']} tokens in "
              f"{result['total_duration']/1e9:.2f}s")

    return state


# ---------------------------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------------------------
graph = StateGraph(ChatState)
graph.add_node("chat", ollama_node)
graph.set_entry_point("chat")
graph.add_edge("chat", END)

# Compile the graph into a callable agent
chat_agent = graph.compile()


# ---------------------------------------------------------------------------
# Example (manual test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Run this file directly to chat in the console:
        python chat_agent.py
    """
    print("🧠 LangGraph Chat Agent (local Ollama backend)")
    print("Type 'exit' or 'quit' to stop.\n")

    state: ChatState = {"messages": [], "context": None}

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        state["messages"].append(user_input)
        result = chat_agent.invoke(state)
        reply = result["messages"][-1]

        print(f"AI: {reply}\n")

        # Update context for next round
        state = result
