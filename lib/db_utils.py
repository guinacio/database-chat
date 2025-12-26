"""Database utilities for the Streamlit Database Chat Agent."""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# Constants - database is in data folder
DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "clientes_completo.db")
DATABASE_URI = f"sqlite:///{DATABASE_PATH}"
MODEL_NAME = "gemini-3-flash-preview"


@lru_cache(maxsize=1)
def get_database() -> SQLDatabase:
    """Create and cache SQLDatabase connection."""
    return SQLDatabase.from_uri(DATABASE_URI)


def get_llm(temperature: float = 0) -> ChatGoogleGenerativeAI:
    """Create the Gemini LLM instance.

    Args:
        temperature: Model temperature (0 for consistent outputs).

    Returns:
        ChatGoogleGenerativeAI instance.

    Raises:
        ValueError: If GOOGLE_API_KEY is not set.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=temperature,
        google_api_key=api_key
    )


def get_toolkit() -> SQLDatabaseToolkit:
    """Create SQLDatabaseToolkit with db and llm."""
    return SQLDatabaseToolkit(db=get_database(), llm=get_llm())


def get_db_schema_context() -> str:
    """Return formatted database schema for prompts."""
    db = get_database()
    tables = db.get_usable_table_names()
    schema_info = []
    for table in tables:
        schema_info.append(db.get_table_info([table]))
    return "\n".join(schema_info)
