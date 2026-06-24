"""
MoodLens REST API — powered by FastAPI

Run:
  uvicorn api:app --reload --port 8000

Then visit: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uuid, time, json
from moodlens import MoodLens

app = FastAPI(
    title="MoodLens API",
    description="AI-powered emotion & sentiment analysis REST API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton analyzer (loads once at startup)
lens = MoodLens(use_transformer=True)


# ── Request / Response models ────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000,
                      example="I can't believe how amazing today turned out to be!")
    use_transformer: bool = Field(True, description="Use deep learning model for higher accuracy")

class CompareRequest(BaseModel):
    text_a: str = Field(..., min_length=1, max_length=10_000)
    text_b: str = Field(..., min_length=1, max_length=10_000)

class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=20)

class AnalyzeResponse(BaseModel):
    request_id: str
    elapsed_ms: float
    result: dict

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "service": "MoodLens API", "version": "1.0.0"}


@app.post("/analyze", response_model=AnalyzeResponse, summary="Analyze a single text")
def analyze(req: AnalyzeRequest):
    """
    Analyze text and return a rich emotion + sentiment profile.

    - **text**: Input text (1–10,000 characters)
    - **use_transformer**: Enable deep-learning emotion model (default: true)
    """
    t0 = time.perf_counter()
    try:
        profile = lens.analyze(req.text)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    return AnalyzeResponse(
        request_id=str(uuid.uuid4()),
        elapsed_ms=elapsed,
        result=profile.to_dict(),
    )


@app.post("/compare", summary="Compare two texts")
def compare(req: CompareRequest):
    """
    Compare the emotional profiles of two texts and return a delta analysis.
    """
    try:
        return lens.compare(req.text_a, req.text_b)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/batch", summary="Analyze up to 20 texts at once")
def batch(req: BatchRequest):
    """
    Batch-analyze a list of texts. Returns one profile per input.
    """
    if len(req.texts) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 texts per batch.")
    try:
        profiles = lens.analyze_batch(req.texts)
        return {"count": len(profiles), "results": [p.to_dict() for p in profiles]}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/emotions", summary="List supported emotions")
def list_emotions():
    from moodlens.analyzer import EMOTION_SEEDS
    return {"emotions": list(EMOTION_SEEDS.keys()),
            "model": "Plutchik Wheel of Emotions + DistilRoBERTa"}
