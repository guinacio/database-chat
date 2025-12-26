"""LangGraph workflow for the Database Chat Agent."""

import base64
import operator
from typing import Annotated, Optional, TypedDict, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from .db_utils import get_llm, get_toolkit

# Visualization marker
NEEDS_VIZ_MARKER = "[NEEDS_VISUALIZATION]"


def get_text_content(content: Union[str, list]) -> str:
    """Extract text content from a message content field.

    Handles both string content and list of content blocks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Extract text from content blocks
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "\n".join(text_parts)
    return str(content)

# System prompts
DB_AGENT_SYSTEM_PROMPT = """You are a SQL expert assistant that helps users query a SQLite database.
The database contains customer data with the following tables:
- clientes: Customer information (id, nome, email, idade, cidade, estado, profissao, genero)
- compras: Purchase records (id, cliente_id, data_compra, valor, categoria, canal)
- campanhas_marketing: Marketing campaigns (id, cliente_id, nome_campanha, data_envio, interagiu, canal)
- suporte: Support tickets (id, cliente_id, data_contato, tipo_contato, resolvido, canal)

IMPORTANT RULES:
1. Use SQLite dialect for all queries
2. Always use the tools to explore the schema before writing queries
3. Use sql_db_query_checker before executing queries when possible
4. Return data in a structured, readable format
5. If the user asks for a visualization, chart, graph, or plot, include [NEEDS_VISUALIZATION] at the END of your response
6. ALWAYS respond in the SAME LANGUAGE as the user's query
7. Limit results to 100 rows unless specifically asked for more
8. Be helpful and explain the results you find
"""

VIZ_AGENT_SYSTEM_PROMPT = """You are a data visualization expert.
You receive query results and generate Python matplotlib code to create visualizations.

IMPORTANT RULES:
1. Generate ONLY valid Python code using matplotlib and pandas
2. The code must create a figure and assign it to a variable named 'fig'
3. Use appropriate chart types (bar, line, pie, scatter, histogram) based on the data
4. Include proper titles, axis labels, and legends
5. Use a clean, professional style with plt.style.use('seaborn-v0_8-whitegrid') or similar
6. ALWAYS respond in the SAME LANGUAGE as the user's query for chart titles and labels
7. Wrap your code in ```python ... ``` markers
8. The data from the query will be provided to you - parse it appropriately
9. Use figsize=(10, 6) for good readability
10. Add colors that are visually appealing
"""


class AgentState(TypedDict):
    """State schema for the LangGraph workflow."""

    messages: Annotated[list[BaseMessage], operator.add]
    query_result: Optional[str]
    visualization_code: Optional[str]
    needs_visualization: bool


def db_agent_node(state: AgentState) -> dict:
    """Process user query using SQLDatabaseToolkit with ReAct pattern."""
    toolkit = get_toolkit()
    llm = get_llm()

    # Create ReAct agent with SQL tools
    agent = create_react_agent(
        llm,
        toolkit.get_tools(),
        prompt=DB_AGENT_SYSTEM_PROMPT
    )

    # Run agent with current messages
    result = agent.invoke({"messages": state["messages"]})

    # Get all messages from the result
    result_messages = result["messages"]

    # Find the last AI message for analysis
    last_ai_message = None
    for msg in reversed(result_messages):
        if isinstance(msg, AIMessage):
            last_ai_message = msg
            break

    # Check if visualization is needed
    needs_viz = False
    query_result = ""
    if last_ai_message:
        query_result = get_text_content(last_ai_message.content)
        needs_viz = NEEDS_VIZ_MARKER in query_result

    return {
        "messages": result_messages[len(state["messages"]):],  # Only new messages
        "query_result": query_result,
        "needs_visualization": needs_viz
    }


def router_node(state: AgentState) -> str:
    """Conditional edge: route to viz_agent or END."""
    if state.get("needs_visualization", False):
        return "viz_agent"
    return "end"


def viz_agent_node(state: AgentState) -> dict:
    """Generate matplotlib visualization code based on query results."""
    llm = get_llm(temperature=0.2)  # Slightly higher temp for creative visualization

    # Get the query result from previous node
    query_result = state.get("query_result", "")

    # Remove the visualization marker
    clean_result = query_result.replace(NEEDS_VIZ_MARKER, "").strip()

    # Create prompt for visualization
    viz_prompt = f"""Based on the following query results, generate Python matplotlib code to create an appropriate visualization.

Query Results:
{clean_result}

Generate complete, executable Python code that:
1. Parses the data from the results above
2. Creates an appropriate chart type for this data
3. Assigns the figure to a variable named 'fig'
4. Uses clear labels and a title in the same language as the data

The code will be executed directly, so make sure it's complete and correct.
"""

    response = llm.invoke([
        {"role": "system", "content": VIZ_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": viz_prompt}
    ])

    # Extract text content from response
    response_text = get_text_content(response.content)

    # Create a combined response message
    combined_response = f"{clean_result}\n\n{response_text}"

    return {
        "messages": [AIMessage(content=combined_response)],
        "visualization_code": response_text,
        "query_result": clean_result  # Update without marker
    }


def process_audio_input(audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
    """Send audio to Gemini multimodal for transcription.

    Args:
        audio_bytes: Raw audio data.
        mime_type: MIME type of the audio (e.g., "audio/wav", "audio/webm").

    Returns:
        Transcribed query text.
    """
    llm = get_llm()

    # Encode audio to base64
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    # Create multimodal message with audio
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Please transcribe this audio message. The user is asking a question "
                    "about a database. Return ONLY the transcribed question, nothing else."
                )
            },
            {
                "type": "media",
                "mime_type": mime_type,
                "data": audio_b64
            }
        ]
    )

    response = llm.invoke([message])
    return get_text_content(response.content).strip()


def create_workflow() -> StateGraph:
    """Build and compile the LangGraph workflow."""
    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("db_agent", db_agent_node)
    workflow.add_node("viz_agent", viz_agent_node)

    # Set entry point
    workflow.set_entry_point("db_agent")

    # Add conditional edge from db_agent
    workflow.add_conditional_edges(
        "db_agent",
        router_node,
        {
            "viz_agent": "viz_agent",
            "end": END
        }
    )

    # viz_agent goes to END
    workflow.add_edge("viz_agent", END)

    # Compile the graph
    return workflow.compile()
