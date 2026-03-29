"""
schemas.py
==========
Pydantic data models for API request and response validation.

FastAPI uses these models to:
    - Validate incoming request data automatically
    - Serialize outgoing response data
    - Generate OpenAPI documentation at /docs

All fields marked Optional can be None in responses
(e.g., when a SMILES is invalid, most fields will be None).
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class PredictRequest(BaseModel):
    """Request body sent by the frontend for a single prediction."""

    smiles: str = Field(
        ...,
        description="SMILES string representing the molecular structure",
        example="CC(=O)Oc1ccccc1C(=O)O"
    )
    compound_name: Optional[str] = Field(
        None,
        description="Optional human-readable name for the compound"
    )


class PredictResponse(BaseModel):
    """
    Response returned by the prediction endpoint.

    Fields:
        valid             — False if the SMILES could not be parsed
        prediction        — "TOXIC" or "NON-TOXIC"
        toxic_probability — 0 to 100, probability the compound is toxic
        safe_probability  — 0 to 100, probability the compound is safe
        risk_level        — "HIGH", "MODERATE", "LOW", or "MINIMAL"
        risk_color        — color hint for frontend UI
        molecular_properties — dict of RDKit-computed descriptors
        toxicity_factors  — list of human-readable risk explanations
        model             — name of the model used for prediction
        error             — error message if valid is False
    """
    valid: bool
    smiles: Optional[str] = None
    compound_name: Optional[str] = None
    prediction: Optional[str] = None
    toxic_probability: Optional[float] = None
    safe_probability: Optional[float] = None
    risk_level: Optional[str] = None
    risk_color: Optional[str] = None
    molecular_properties: Optional[Dict[str, Any]] = None
    toxicity_factors: Optional[List[str]] = None
    model: Optional[str] = None
    error: Optional[str] = None


class BatchPredictRequest(BaseModel):
    """Request body for batch (multi-compound) prediction."""
    compounds: List[PredictRequest]


class BatchPredictResponse(BaseModel):
    """
    Response for batch prediction with summary statistics.

    Fields:
        results     — list of individual prediction results
        total       — total number of compounds analyzed
        toxic_count — number predicted as TOXIC
        safe_count  — number predicted as NON-TOXIC
    """
    results: List[PredictResponse]
    total: int
    toxic_count: int
    safe_count: int