# app/config.py
import os
from dotenv import load_dotenv

# Load environment variables from the .env file at the root of the project
load_dotenv()

# --- App Configuration ---
class Config:
    """Holds all application configuration."""
    # Model and Auth
    MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-2-7b-chat-hf")
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    DEVICE = os.getenv("DEVICE", "auto")

    # APIs
    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
    API_URL = os.getenv("API_URL", "http://localhost:8000/classify")