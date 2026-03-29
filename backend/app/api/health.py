"""
health.py
=========
Health check endpoint for the API server.

Returns the current status of:
    - RDKit installation (required for feature extraction)
    - ML model availability (required for predictions)

Frontend uses this endpoint on page load to show
the "MODEL READY" or "SERVER DOWN" status indicator.
"""

from fastapi import APIRouter
import os

router = APIRouter()


@router.get("/health")
async def health():
    """
    Returns server health status.

    Checks:
        rdkit_available — whether RDKit is installed and importable
        model_trained   — whether toxicity_model.pkl exists on disk
        message         — human-readable status string
    """
    # Build path to the trained model file
    model_path = os.path.join(
        os.path.dirname(__file__),
        "../../ml_model/toxicity_model.pkl"
    )
    model_exists = os.path.exists(model_path)

    # Try importing RDKit to confirm it is installed
    try:
        from rdkit import Chem
        rdkit_ok = True
    except ImportError:
        rdkit_ok = False

    # Generate human-readable status message
    if rdkit_ok and model_exists:
        message = "Ready"
    elif not rdkit_ok:
        message = "RDKit not installed — run: pip install rdkit"
    else:
        message = "Model not trained — run: python train_model.py"

    return {
        "status":          "ok",
        "rdkit_available": rdkit_ok,
        "model_trained":   model_exists,
        "message":         message
    }