import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Data Files
RAW_DATA_PATH = DATA_DIR / "anime_with_synopsis.csv"
PROCESSED_DATA_PATH = DATA_DIR / "anime_updated.csv"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

# Fallback path if data is under expreiments/
if not RAW_DATA_PATH.exists():
    ALT_RAW = BASE_DIR / "expreiments" / "anime_with_synopsis.csv"
    if ALT_RAW.exists():
        RAW_DATA_PATH = ALT_RAW
        PROCESSED_DATA_PATH = BASE_DIR / "expreiments" / "anime_updated.csv"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# LLM Configuration
MODEL_PROVIDER = "groq"
MODEL_NAME = "openai/gpt-oss-20b"
TEMPERATURE = 0.2
MAX_TOKENS = 600

# Embedding & Vector Store Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVER_TOP_K = 5
