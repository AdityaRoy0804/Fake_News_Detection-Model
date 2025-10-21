# app/classifier.py
import os
import re
import json
import logging
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from app.prompt_templates import build_prompt
from app.source_search import search_newsapi

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Environment variables
MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-2-7b-instruct")
DEVICE = os.getenv("DEVICE", "auto")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # loaded from .env

if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found. Please set it in your .env file.")

class LlamaVerifier:
    def __init__(self, model_id: str = MODEL_ID, device: str = DEVICE, use_few_shot: bool = True):
        self.model_id = model_id
        self.device = device
        self.use_few_shot = use_few_shot
        self._load_model()

    def _load_model(self):
        logger.info(f"Loading model {self.model_id} on device={self.device} ...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, use_fast=True, use_auth_token=HF_TOKEN
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            use_auth_token=HF_TOKEN
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.model.device.type == "cuda" else -1
        )
        logger.info("Model loaded successfully.")

    def _post_process_json(self, raw_text: str) -> Dict[str, Any]:
        m = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not m:
            try:
                return json.loads(raw_text)
            except Exception:
                return {"label": "UNKNOWN", "confidence": 0.0, "explanation": raw_text, "sources": []}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {"label": "UNKNOWN", "confidence": 0.0, "explanation": raw_text, "sources": []}

    def classify(self, news_text: str) -> Dict[str, Any]:
        prompt = build_prompt(news_text, use_few_shot=self.use_few_shot)
        out = self.pipe(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]
        parsed = self._post_process_json(out)
        if not parsed.get("sources"):
            try:
                parsed["sources"] = search_newsapi(news_text, page_size=3)
            except Exception:
                pass
        return parsed

_verifier = None

def get_verifier():
    global _verifier
    if _verifier is None:
        _verifier = LlamaVerifier()
    return _verifier
