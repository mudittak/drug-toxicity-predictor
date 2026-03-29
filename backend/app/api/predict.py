"""
predict.py
==========
API route handlers for drug toxicity prediction.

Endpoints:
    POST /api/predict         — predict toxicity of a single molecule
    POST /api/predict/batch   — predict toxicity for multiple molecules
    GET  /api/examples        — returns example SMILES for testing
    GET  /api/molecule-image  — returns 2D structure image as PNG
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from app.models.schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse
)
from app.services.predict_service import predict_toxicity
import io

# Create router — all routes defined here will be registered
# in main.py under the /api prefix
router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict toxicity of a single compound.

    Input:
        smiles (str)         — SMILES string of the molecule
        compound_name (str)  — optional human-readable name

    Output:
        PredictResponse with prediction, probability,
        risk level, molecular properties, and key factors.
    """
    try:
        result = predict_toxicity(request.smiles.strip())

        # Attach compound name to result if user provided one
        if request.compound_name:
            result["compound_name"] = request.compound_name

        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """
    Predict toxicity for multiple compounds at once.

    Processes each compound individually. If one fails,
    it returns an error result for that compound only
    and continues with the rest.

    Input:
        compounds — list of PredictRequest objects

    Output:
        BatchPredictResponse with all results + summary counts
    """
    results = []

    for compound in request.compounds:
        try:
            result = predict_toxicity(compound.smiles.strip())
            if compound.compound_name:
                result["compound_name"] = compound.compound_name
            results.append(PredictResponse(**result))
        except Exception as e:
            # Don't fail the whole batch — record the error and continue
            results.append(PredictResponse(valid=False, error=str(e)))

    # Count toxic vs safe for the summary card
    toxic_count = sum(1 for r in results if r.prediction == "TOXIC")
    safe_count  = sum(1 for r in results if r.prediction == "NON-TOXIC")

    return BatchPredictResponse(
        results=results,
        total=len(results),
        toxic_count=toxic_count,
        safe_count=safe_count
    )


@router.get("/examples")
async def get_examples():
    """
    Returns a list of example SMILES strings for testing.
    These cover all risk levels from MINIMAL to HIGH.
    """
    return {
        "examples": [
            {"name": "Aspirin",              "smiles": "CC(=O)Oc1ccccc1C(=O)O",      "expected": "Low-Moderate Risk"},
            {"name": "Naphthalene",          "smiles": "c1ccc2ccccc2c1",              "expected": "Moderate Risk"},
            {"name": "Pyrene (PAH)",         "smiles": "c1cc2ccc3cccc4ccc(c1)c2c34", "expected": "High Risk"},
            {"name": "Ethanol",              "smiles": "CCO",                         "expected": "Minimal Risk"},
            {"name": "Glycerol",             "smiles": "OCC(O)CO",                    "expected": "Minimal Risk"},
            {"name": "Nitrobenzene",         "smiles": "c1ccc(cc1)N(=O)=O",          "expected": "High Risk"},
            {"name": "Carbon Tetrachloride", "smiles": "ClC(Cl)(Cl)Cl",              "expected": "High Risk"},
            {"name": "Acetic Acid",          "smiles": "CC(=O)O",                    "expected": "Minimal Risk"},
        ]
    }


@router.get("/molecule-image/{smiles:path}")
async def molecule_image(smiles: str):
    """
    Generate and return a 2D structure image of a molecule.

    Uses RDKit's Draw module to render the molecule as a PNG image.
    The image is returned as binary content with media type image/png.

    Input:
        smiles (str) — URL-encoded SMILES string (in path)

    Output:
        PNG image of the molecular structure (300x250 px)
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw

        # Parse SMILES into RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise HTTPException(status_code=400, detail="Invalid SMILES string")

        # Render molecule as PIL image
        img = Draw.MolToImage(mol, size=(300, 250))

        # Convert PIL image to PNG bytes for HTTP response
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))