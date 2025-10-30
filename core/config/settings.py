import os
from pathlib import Path
from dotenv import load_dotenv

# LOAD ENVIRONMENT VARIABLES
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(os.path.join(BASE_DIR, '.env'))

# GENERAL SETTINGS
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
DEBUG = ENVIRONMENT == 'development'

# LLM / LANGCHAIN SETTINGS
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'huggingface')
API_KEYS = {
    'huggingface': os.getenv('HUGGINGFACE_API_KEY'),
}

# DATABASE SETTINGS
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/data/core.db")

# LOGGING SETTINGS
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "logs/app.log"

