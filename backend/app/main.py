"""
main.py
=======
FastAPI application entry point.

This file initializes the web server, sets up CORS middleware
so the React frontend can talk to this backend, and registers
all API route handlers (predict, health).

Run with:
    python -m uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import predict, health

# Initialize the FastAPI application with metadata
app = FastAPI(
    title="Drug Toxicity Predictor API",
    description="ML-based drug toxicity prediction using molecular descriptors",
    version="1.0.0"
)

# ── CORS Middleware ────────────────────────────────────────────
# Allows the React frontend (running on port 5173) to make
# HTTP requests to this backend (running on port 8000).
# In production, replace "*" with your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all origins (dev only)
    allow_credentials=True,
    allow_methods=["*"],       # Allow GET, POST, etc.
    allow_headers=["*"],       # Allow all headers
)

# ── Register Routers ───────────────────────────────────────────
# Each router handles a group of related API endpoints.
# All endpoints are prefixed with /api
app.include_router(predict.router, prefix="/api", tags=["prediction"])
app.include_router(health.router,  prefix="/api", tags=["health"])


@app.get("/")
def root():
    """Root endpoint — confirms the API is running."""
    return {"message": "Drug Toxicity Predictor API is running"}