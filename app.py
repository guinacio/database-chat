"""Streamlit Database Chat Agent Application."""

import ast
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from lib import create_workflow, process_audio_input, get_text_content

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Database Chat Agent",
    page_icon=":speech_balloon:",
    layout="wide"
)


def extract_python_code(text: str) -> str | None:
    """Extract Python code from markdown code blocks."""
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0].strip() if matches else None


def strip_imports(code: str) -> str:
    """Remove import statements from code since we provide modules."""
    lines = code.split("\n")
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip import lines
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines)


def execute_visualization(code: str) -> plt.Figure | None:
    """Execute matplotlib code and return the figure.

    Args:
        code: Python code that creates a matplotlib figure.

    Returns:
        The matplotlib figure, or None if execution failed.
    """
    # Strip import statements since we provide the modules
    code = strip_imports(code)

    # Create execution context with necessary imports
    local_vars = {
        # Pre-define common variables that LLM might use
        "width": 10,
        "height": 6,
        "figsize": (10, 6),
    }
    global_vars = {
        "pd": pd,
        "plt": plt,
        "np": np,
        "matplotlib": matplotlib,
        "__builtins__": {
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "sorted": sorted,
            "reversed": reversed,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "print": print,
            "isinstance": isinstance,
            "type": type,
            "map": map,
            "filter": filter,
            "set": set,
            "frozenset": frozenset,
        }
    }

    try:
        # Set default figure size before execution
        plt.rcParams['figure.figsize'] = [10, 6]

        exec(code, global_vars, local_vars)
        fig = local_vars.get("fig")
        if fig is None:
            # Try to get current figure if 'fig' wasn't explicitly assigned
            fig = plt.gcf()

        # Enforce maximum figure size
        if fig is not None:
            fig.set_size_inches(10, 6)

        return fig
    except Exception as e:
        st.error(f"Error executing visualization: {e}")
        return None


def is_tabular_data(text: str) -> bool:
    """Heuristic to detect if response contains tabular data."""
    # Check for list of tuples/lists pattern from SQL results
    if text.strip().startswith("[") and "]," in text:
        return True
    # Check for markdown table pattern
    if re.search(r"\|.*\|.*\|", text):
        return True
    return False


def parse_to_dataframe(text: str) -> pd.DataFrame | None:
    """Attempt to parse text response to DataFrame."""
    try:
        # Handle list of tuples format from SQL
        if text.strip().startswith("["):
            # Find the list in the text
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                data = ast.literal_eval(match.group())
                if data and isinstance(data[0], (tuple, list)):
                    return pd.DataFrame(data)
    except Exception:
        pass
    return None


def render_response(result: dict):
    """Render the agent response appropriately."""
    visualization_code = result.get("visualization_code")
    query_result = result.get("query_result", "")

    # Handle visualization if present
    if visualization_code:
        code = extract_python_code(visualization_code)
        if code:
            fig = execute_visualization(code)
            if fig:
                st.pyplot(fig, width='content')
                plt.close(fig)

        # Show the text response (without viz code)
        # Remove the code block from display
        display_text = re.sub(r"```python.*?```", "", query_result, flags=re.DOTALL)
        if display_text.strip():
            st.markdown(display_text.strip())
    else:
        # Check for tabular data
        if is_tabular_data(query_result):
            df = parse_to_dataframe(query_result)
            if df is not None:
                st.dataframe(df, width='stretch')
                # Also show any text before/after the data
                text_parts = re.split(r"\[.*\]", query_result, flags=re.DOTALL)
                for part in text_parts:
                    if part.strip():
                        st.markdown(part.strip())
            else:
                st.markdown(query_result)
        else:
            st.markdown(query_result)


def extract_status_info(messages: list) -> dict:
    """Extract reasoning and SQL queries from agent messages.

    Returns a dict with:
    - reasoning: List of reasoning steps
    - sql_queries: List of SQL queries executed
    - tool_results: List of tool results
    """
    reasoning = []
    sql_queries = []
    tool_results = []

    for msg in messages:
        if isinstance(msg, AIMessage):
            # Check for tool calls (SQL queries)
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call.get('name') == 'sql_db_query':
                        query = tool_call.get('args', {}).get('query', '')
                        if query:
                            sql_queries.append(query)
            # Extract reasoning from content
            content = get_text_content(msg.content)
            if content and not content.startswith('['):  # Skip marker-only messages
                # Only add if it looks like reasoning (not just the final answer)
                if len(content) < 500:  # Short messages are likely reasoning
                    reasoning.append(content)

        elif isinstance(msg, ToolMessage):
            # Capture tool results
            content = get_text_content(msg.content) if hasattr(msg, 'content') else str(msg)
            if content and len(content) < 1000:  # Truncate long results
                tool_results.append(content[:500] + "..." if len(content) > 500 else content)

    return {
        "reasoning": reasoning,
        "sql_queries": sql_queries,
        "tool_results": tool_results
    }


def render_status_details(status_info: dict):
    """Render the status details inside an st.status container."""
    if status_info.get("sql_queries"):
        st.markdown("**SQL Queries:**")
        for i, query in enumerate(status_info["sql_queries"], 1):
            st.code(query, language="sql")


def main():
    """Main application entry point."""
    st.title(":speech_balloon: Database Chat Agent")
    st.caption("Ask questions about your customer database using natural language or voice")

    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")

        # Check if API key is already loaded from .env
        existing_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value="" if existing_key else "",
            help="Enter your Google Gemini API key (or set GEMINI_API_KEY in .env)",
            placeholder="Loaded from .env" if existing_key else "Enter API key..."
        )
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key

        st.divider()

        st.subheader("Available Tables")
        st.markdown("""
        - **clientes** - Customer information
          - nome, email, idade, cidade, estado, profissao, genero
        - **compras** - Purchase records
          - data_compra, valor, categoria, canal
        - **campanhas_marketing** - Marketing campaigns
          - nome_campanha, data_envio, interagiu, canal
        - **suporte** - Support tickets
          - data_contato, tipo_contato, resolvido, canal
        """)

        st.divider()

        st.subheader("Example Questions")
        st.markdown("""
        - "List the first 5 customers"
        - "How many purchases were made?"
        - "Show sales by category as a bar chart"
        - "What's the average customer age by city?"
        - "Quantos clientes existem por estado?"
        """)

        if st.button("Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "workflow" not in st.session_state:
        st.session_state.workflow = None

    # Check for API key
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        st.warning("Please enter your Gemini API Key in the sidebar to get started, or set GEMINI_API_KEY in .env file.")
        return

    # Initialize workflow if needed
    if st.session_state.workflow is None:
        try:
            st.session_state.workflow = create_workflow()
        except Exception as e:
            st.error(f"Error initializing workflow: {e}")
            return

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Show status info if available (collapsed for history)
            if message.get("status_info") and message["status_info"].get("sql_queries"):
                with st.status("Query complete", state="complete", expanded=False):
                    render_status_details(message["status_info"])

            # Re-render visualization if we have stored code
            if message.get("visualization_code"):
                code = extract_python_code(message["visualization_code"])
                if code:
                    fig = execute_visualization(code)
                    if fig:
                        st.pyplot(fig, width='content')
                        plt.close(fig)
            if message.get("dataframe") is not None:
                st.dataframe(message["dataframe"], width='stretch')
            if message.get("content"):
                st.markdown(message["content"])

    # Chat input with audio support
    user_input = st.chat_input(
        "Ask a question about the database...",
        accept_audio=True
    )

    # Process input
    query = None

    if user_input:
        # Handle text input
        if user_input.text:
            query = user_input.text
        # Handle audio input
        elif user_input.audio:
            with st.spinner("Transcribing audio..."):
                try:
                    audio_bytes = user_input.audio.read()
                    if audio_bytes and len(audio_bytes) > 100:
                        # Browser audio recording is typically webm format
                        query = process_audio_input(audio_bytes, "audio/webm")
                        if query and query.strip():
                            st.toast(f"🎤 Transcribed: {query[:50]}..." if len(query) > 50 else f"🎤 Transcribed: {query}")
                        else:
                            st.warning("Could not transcribe audio. Please try again or type your question.")
                            query = None
                    else:
                        st.warning("Audio recording was too short or empty. Please try again.")
                        query = None
                except Exception as e:
                    st.error(f"Error processing audio: {e}")
                    query = None

    if query:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Process with LangGraph workflow
        with st.chat_message("assistant"):
            try:
                # Use st.status to show processing steps
                with st.status("Processing query...", expanded=True) as status:
                    status.update(label="Analyzing question...", state="running")

                    # Invoke the workflow
                    result = st.session_state.workflow.invoke({
                        "messages": [HumanMessage(content=query)],
                        "needs_visualization": False,
                        "query_result": None,
                        "visualization_code": None
                    })

                    # Extract and display status info
                    status_info = extract_status_info(result.get("messages", []))

                    # Show SQL queries if any
                    if status_info["sql_queries"]:
                        status.update(label="Executing SQL...", state="running")
                        render_status_details(status_info)

                    # Mark as complete
                    status.update(label="Query complete", state="complete", expanded=False)

                # Render the response
                render_response(result)

                # Store in history (include visualization code and status info for re-rendering)
                visualization_code = result.get("visualization_code")
                query_result = result.get("query_result", "")

                # Clean the display text if there's visualization code
                if visualization_code:
                    display_text = re.sub(r"```python.*?```", "", query_result, flags=re.DOTALL).strip()
                else:
                    display_text = query_result

                assistant_message = {
                    "role": "assistant",
                    "content": display_text,
                    "visualization_code": visualization_code,
                    "status_info": status_info
                }

                st.session_state.messages.append(assistant_message)

            except Exception as e:
                error_msg = f"Error processing query: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

            # Rerun to redraw from history and position input bar at bottom
            st.rerun()


if __name__ == "__main__":
    main()
