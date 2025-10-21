from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from .classifier import get_verifier
from .utils import load_env


# load .env
load_env()


app = FastAPI(title="LLaMA Fake News Verifier")


class NewsItem(BaseModel):
    text: str
    id: Optional[str] = None


@app.post("/classify")
async def classify_item(item: NewsItem):
    verifier = get_verifier()
    try:
        result = verifier.classify(item.text)
        return {"id": item.id, "input": item.text, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# simple health route
@app.get("/health")
def health():
    return {"status": "ok"}