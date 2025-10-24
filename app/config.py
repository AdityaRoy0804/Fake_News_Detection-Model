# app/config.py
import os
from dotenv import load_dotenv

# --- App Configuration ---
class Config:
    """Holds all application configuration."""
    def __init__(self):
        # Load environment variables from the .env file at the root of the project
        load_dotenv(override=True) # Ensure .env values override existing ones
        # Model and Auth
        self.MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-2-7b-chat-hf")
        self.HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
        self.DEVICE = os.getenv("DEVICE", "auto")
        # APIs
        self.NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
        self.API_URL = os.getenv("API_URL", "http://localhost:8000/classify")

        # Debug: Print the loaded Hugging Face token
        print(f"Hugging Face Token loaded: {self.HF_TOKEN}")

# Create a single, global instance of the configuration
app_config = Config()