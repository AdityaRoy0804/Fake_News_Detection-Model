# app/classifier.py
import os
import re
import json
import logging
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from app.config import Config
from app.prompt_templates import build_prompt
from app.source_search import search_newsapi

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class LlamaVerifier:
    def __init__(self, config: Config, use_few_shot: bool = True):
        self.config = config
        self.model_id = self.config.MODEL_ID
        self.hf_token = self.config.HF_TOKEN
        self.use_few_shot = use_few_shot
        self._load_model()


    def _load_model(self):
        logger.info(f"Loading model {self.model_id}...")
        if not self.hf_token:
            raise ValueError("Hugging Face token is missing. Cannot load model.")
        
        # For CPU, loading in 8-bit is highly recommended to save RAM and improve speed.
        # This requires the `bitsandbytes` library.
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, use_fast=True, token=self.config.HF_TOKEN
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=quantization_config,
            token=self.config.HF_TOKEN
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
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
                parsed["sources"] = search_newsapi(news_text, api_key=self.config.NEWSAPI_KEY, page_size=3)
            except Exception:
                pass
        return parsed

_verifier = None

def get_verifier():
    global _verifier
    if _verifier is None:
        config = Config()
        logger.info(f"Initializing verifier with token: {'Yes' if config.HF_TOKEN else 'No'}")
        _verifier = LlamaVerifier(config=config)
    return _verifier
