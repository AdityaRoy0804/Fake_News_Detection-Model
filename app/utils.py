# app/utils.py
import os
import time
import logging
from functools import lru_cache


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llama_verifier")


@lru_cache(maxsize=128)
def cached_query_hash(text: str) -> str:
    # simple cache key
    return str(abs(hash(text)))




def load_env():
    # simple wrapper to load .env if present
    from dotenv import load_dotenv
    load_dotenv()