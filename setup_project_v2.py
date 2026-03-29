"""
ToxPredict - Auto Project Setup Script
Run this from E:\\drug-toxicity-predictor
Command: python setup_project.py
"""

import os

def write(path, content):
    folder = os.path.dirname(path)
    if folder and folder != ".":
        os.makedirs(folder, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  Created: {path}")

print("\n ToxPredict - Project Setup")
print("=" * 40)

# ─── BACKEND FILES ───────────────────────────────────────────

write("backend/app/__init__.py", "")
write("backend/app/api/__init__.py", "")
write("backend/app/models/__init__.py", "")
write("backend/app/services/__init__.py", "")

write("backend/app/main.py", '''from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import predict, health

app = FastAPI(
    title="Drug Toxicity Predictor API",
    description="ML-based drug toxicity prediction using molecular descriptors",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api", tags=["prediction"])
app.include_router(health.router, prefix="/api", tags=["health"])

@app.get("/")
def root():
    return {"message": "Drug Toxicity Predictor API is running"}
''')

write("backend/app/api/predict.py", '''from fastapi import APIRouter, HTTPException
from app.models.schemas import PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse
from app.services.predict_service import predict_toxicity

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict toxicity of a single compound from its SMILES string"""
    try:
        result = predict_toxicity(request.smiles.strip())
        if request.compound_name:
            result["compound_name"] = request.compound_name
        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """Predict toxicity for multiple compounds at once"""
    results = []
    for compound in request.compounds:
        try:
            result = predict_toxicity(compound.smiles.strip())
            if compound.compound_name:
                result["compound_name"] = compound.compound_name
            results.append(PredictResponse(**result))
        except Exception as e:
            results.append(PredictResponse(valid=False, error=str(e)))

    toxic_count = sum(1 for r in results if r.prediction == "TOXIC")
    safe_count = sum(1 for r in results if r.prediction == "NON-TOXIC")

    return BatchPredictResponse(
        results=results,
        total=len(results),
        toxic_count=toxic_count,
        safe_count=safe_count
    )

@router.get("/examples")
async def get_examples():
    return {
        "examples": [
            {"name": "Aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "expected": "Low-Moderate Risk"},
            {"name": "Naphthalene", "smiles": "c1ccc2ccccc2c1", "expected": "Moderate Risk"},
            {"name": "Pyrene (PAH)", "smiles": "c1cc2ccc3cccc4ccc(c1)c2c34", "expected": "High Risk"},
            {"name": "Ethanol", "smiles": "CCO", "expected": "Minimal Risk"},
            {"name": "Glycerol", "smiles": "OCC(O)CO", "expected": "Minimal Risk"},
            {"name": "Nitrobenzene", "smiles": "c1ccc(cc1)N(=O)=O", "expected": "High Risk"},
            {"name": "Carbon Tetrachloride", "smiles": "ClC(Cl)(Cl)Cl", "expected": "High Risk"},
            {"name": "Acetic Acid", "smiles": "CC(=O)O", "expected": "Minimal Risk"},
        ]
    }
''')

write("backend/app/api/health.py", '''from fastapi import APIRouter
import os

router = APIRouter()

@router.get("/health")
async def health():
    model_exists = os.path.exists(
        os.path.join(os.path.dirname(__file__), "../../ml_model/toxicity_model.pkl")
    )
    try:
        from rdkit import Chem
        rdkit_ok = True
    except ImportError:
        rdkit_ok = False

    return {
        "status": "ok",
        "rdkit_available": rdkit_ok,
        "model_trained": model_exists,
        "message": "Ready" if (rdkit_ok and model_exists) else
                   "RDKit missing" if not rdkit_ok else
                   "Model not trained — run train_model.py"
    }
''')

write("backend/app/models/schemas.py", '''from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class PredictRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the molecule", example="CC(=O)Oc1ccccc1C(=O)O")
    compound_name: Optional[str] = Field(None, description="Optional name of the compound")

class MolecularProperties(BaseModel):
    molecular_weight: float
    logP: float
    h_bond_donors: int
    h_bond_acceptors: int
    tpsa: float
    rotatable_bonds: int
    aromatic_rings: int
    heavy_atom_count: int
    lipinski_violations: int
    drug_likeness: str

class PredictResponse(BaseModel):
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
    compounds: List[PredictRequest]

class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total: int
    toxic_count: int
    safe_count: int
''')

write("backend/app/services/predict_service.py", '''import pickle
import os
import numpy as np
from typing import Optional

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../../ml_model/toxicity_model.pkl")
INFO_PATH = os.path.join(os.path.dirname(__file__), "../../../ml_model/model_info.pkl")

_model = None
_model_info = None

def load_model():
    global _model, _model_info
    if _model is None:
        try:
            with open(MODEL_PATH, "rb") as f:
                _model = pickle.load(f)
            with open(INFO_PATH, "rb") as f:
                _model_info = pickle.load(f)
            print("ML model loaded successfully")
        except FileNotFoundError:
            print("Model not found — run train_model.py first")
    return _model, _model_info

def validate_smiles(smiles: str) -> bool:
    if not RDKIT_AVAILABLE:
        return True
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def extract_features(smiles: str):
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_list = list(fp)
    descriptors = [
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol), mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcNumRings(mol), Descriptors.FractionCSP3(mol),
        Descriptors.BalabanJ(mol) if mol.GetNumBonds() > 0 else 0,
        Descriptors.BertzCT(mol), Descriptors.Chi0(mol), Descriptors.Chi1(mol),
        Descriptors.HallKierAlpha(mol), Descriptors.Kappa1(mol),
        Descriptors.Kappa2(mol), Descriptors.LabuteASA(mol),
        Descriptors.MaxPartialCharge(mol), Descriptors.MinPartialCharge(mol),
    ]
    return fp_list + descriptors

def get_molecular_properties(smiles: str) -> dict:
    if not RDKIT_AVAILABLE:
        return {}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rotbonds = Descriptors.NumRotatableBonds(mol)
    arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    heavy = mol.GetNumHeavyAtoms()
    lipinski_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return {
        "molecular_weight": round(mw, 2), "logP": round(logp, 3),
        "h_bond_donors": int(hbd), "h_bond_acceptors": int(hba),
        "tpsa": round(tpsa, 2), "rotatable_bonds": int(rotbonds),
        "aromatic_rings": int(arom_rings), "heavy_atom_count": int(heavy),
        "lipinski_violations": lipinski_violations,
        "drug_likeness": "Drug-like" if lipinski_violations <= 1 else "Poor drug-likeness"
    }

def predict_toxicity(smiles: str) -> dict:
    model, model_info = load_model()
    if not validate_smiles(smiles):
        return {"error": "Invalid SMILES string.", "valid": False}
    features = extract_features(smiles)
    if features is None:
        return {"error": "Could not extract features.", "valid": False}
    if model is None:
        props = get_molecular_properties(smiles)
        return rule_based_prediction(smiles, props)
    features_arr = np.array(features).reshape(1, -1)
    prediction = model.predict(features_arr)[0]
    probability = model.predict_proba(features_arr)[0]
    toxic_prob = float(probability[1])
    safe_prob = float(probability[0])
    if toxic_prob >= 0.75:
        risk_level, risk_color = "HIGH", "red"
    elif toxic_prob >= 0.45:
        risk_level, risk_color = "MODERATE", "orange"
    elif toxic_prob >= 0.25:
        risk_level, risk_color = "LOW", "yellow"
    else:
        risk_level, risk_color = "MINIMAL", "green"
    props = get_molecular_properties(smiles)
    toxicity_factors = []
    if props.get("aromatic_rings", 0) >= 3:
        toxicity_factors.append("Multiple aromatic rings (PAH-like structure)")
    if props.get("logP", 0) > 5:
        toxicity_factors.append("High lipophilicity (LogP > 5) — bioaccumulation risk")
    if props.get("molecular_weight", 0) > 500:
        toxicity_factors.append("High molecular weight — poor bioavailability")
    if props.get("tpsa", 0) < 20:
        toxicity_factors.append("Low TPSA — high membrane permeability")
    if not toxicity_factors:
        toxicity_factors.append("No major structural alerts detected")
    return {
        "valid": True, "smiles": smiles,
        "prediction": "TOXIC" if prediction == 1 else "NON-TOXIC",
        "toxic_probability": round(toxic_prob * 100, 1),
        "safe_probability": round(safe_prob * 100, 1),
        "risk_level": risk_level, "risk_color": risk_color,
        "molecular_properties": props,
        "toxicity_factors": toxicity_factors,
        "model": "Random Forest (Morgan Fingerprints + RDKit Descriptors)"
    }

def rule_based_prediction(smiles: str, props: dict) -> dict:
    score = 0
    factors = []
    if props.get("aromatic_rings", 0) >= 2:
        score += 0.3; factors.append("Multiple aromatic rings detected")
    if props.get("logP", 0) > 4:
        score += 0.2; factors.append("High lipophilicity")
    if props.get("molecular_weight", 0) > 400:
        score += 0.1; factors.append("High molecular weight")
    if "Cl" in smiles or "Br" in smiles:
        score += 0.3; factors.append("Halogen atoms present")
    if "N(=O)" in smiles or "[N+](=O)" in smiles:
        score += 0.25; factors.append("Nitro group detected")
    score = min(score, 1.0)
    risk = "HIGH" if score > 0.6 else "MODERATE" if score > 0.3 else "LOW"
    return {
        "valid": True, "smiles": smiles,
        "prediction": "TOXIC" if score > 0.4 else "NON-TOXIC",
        "toxic_probability": round(score * 100, 1),
        "safe_probability": round((1 - score) * 100, 1),
        "risk_level": risk,
        "risk_color": "red" if risk == "HIGH" else "orange" if risk == "MODERATE" else "green",
        "molecular_properties": props,
        "toxicity_factors": factors if factors else ["No major alerts"],
        "model": "Rule-based (ML model not trained yet)"
    }
''')

write("backend/ml_model/train_model.py", '''import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

TOXIC_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O","c1ccc2ccccc2c1","C1=CC=C(C=C1)N",
    "ClC1=CC=CC=C1","BrC1=CC=CC=C1","Clc1ccc(Cl)cc1",
    "c1ccc(cc1)N(=O)=O","C(Cl)(Cl)(Cl)Cl","ClCC(Cl)Cl",
    "c1ccncc1","c1ccc(cc1)Cl","CC(N)=O","N#Cc1ccccc1",
    "O=C1NC(=O)NC1=O","c1ccc2[nH]ccc2c1","O=C(O)c1ccccc1O",
    "C1=CC=NC=C1","O=C(c1ccccc1)c1ccccc1","c1ccc(-c2ccccc2)cc1",
    "c1ccc2c(c1)ccc1ccccc12","c1cc2ccc3cccc4ccc(c1)c2c34",
    "ClCCl","BrCBr","O=C1OCC1","C1CO1",
]

SAFE_SMILES = [
    "CCO","CC(C)O","OCC(O)CO","OC1CCCCC1","CC(=O)O",
    "OC(=O)C(O)=O","OC(CO)(CO)CO","OCC(O)C(O)C(O)C(O)CO",
    "CC1=CC=CC=C1","CCCO","CCCCO","OC1=CC=CC=C1",
    "C1CCCCC1","CCCCCC","C1CCCC1","CC(C)=O",
    "CCOC(C)=O","COC(=O)C","CCOCC","C1CCCOC1",
    "NCC(=O)O","CC(N)C(=O)O","OC(Cc1ccccc1)C(=O)O","CC(C)(N)C(=O)O",
]

def get_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = list(GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    desc = [
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol), mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcNumRings(mol), Descriptors.FractionCSP3(mol),
        Descriptors.BalabanJ(mol) if mol.GetNumBonds() > 0 else 0,
        Descriptors.BertzCT(mol), Descriptors.Chi0(mol), Descriptors.Chi1(mol),
        Descriptors.HallKierAlpha(mol), Descriptors.Kappa1(mol),
        Descriptors.Kappa2(mol), Descriptors.LabuteASA(mol),
        Descriptors.MaxPartialCharge(mol), Descriptors.MinPartialCharge(mol),
    ]
    return fp + desc

if __name__ == "__main__":
    if not RDKIT_AVAILABLE:
        print("ERROR: Install rdkit first: pip install rdkit")
        exit()

    print("Generating features...")
    X, y = [], []
    for smi in TOXIC_SMILES:
        f = get_features(smi)
        if f: X.append(f); y.append(1)
    for smi in SAFE_SMILES:
        f = get_features(smi)
        if f: X.append(f); y.append(0)

    X, y = np.array(X), np.array(y)
    print(f"Dataset: {len(X)} compounds ({sum(y)} toxic, {len(y)-sum(y)} safe)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    print("Training...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Toxic"]))

    os.makedirs(".", exist_ok=True)
    with open("toxicity_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("model_info.pkl", "wb") as f:
        pickle.dump({"n_features": X.shape[1]}, f)

    print("Model saved! Training complete!")
''')

write("backend/requirements.txt",
"""fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.7.1
rdkit==2024.3.1
scikit-learn==1.5.0
numpy==1.26.4
pandas==2.2.2
python-multipart==0.0.9
""")

# ─── FRONTEND FILES ──────────────────────────────────────────

write("frontend/index.html", """<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>ToxPredict — Drug Toxicity Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
""")

write("frontend/package.json", """{
  "name": "drug-toxicity-frontend",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "axios": "^1.7.2",
    "lucide-react": "^0.383.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.0",
    "autoprefixer": "^10.4.19",
    "postcss": "^8.4.38",
    "tailwindcss": "^3.4.4",
    "vite": "^5.2.13"
  }
}
""")

write("frontend/vite.config.js", """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true }
    }
  }
})
""")

write("frontend/tailwind.config.js", """/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        mono: ["Space Mono", "monospace"],
        sans: ["DM Sans", "sans-serif"],
      },
      colors: {
        bg: "#050a0e", surface: "#0d1117", card: "#111827",
        border: "#1f2937", accent: "#00d4aa", danger: "#ff4d6d",
        warn: "#ffb703", safe: "#06d6a0", muted: "#6b7280",
      },
    },
  },
  plugins: [],
}
""")

write("frontend/postcss.config.js", """export default {
  plugins: { tailwindcss: {}, autoprefixer: {} }
}
""")

write("frontend/src/main.jsx", """import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode><App /></React.StrictMode>
)
""")

write("frontend/src/index.css", """@tailwind base;
@tailwind components;
@tailwind utilities;
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background-color: #050a0e; color: #e5e7eb; font-family: "DM Sans", sans-serif; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #1f2937; border-radius: 3px; }
.glow-accent { box-shadow: 0 0 20px rgba(0,212,170,0.15); }
.glow-danger { box-shadow: 0 0 20px rgba(255,77,109,0.2); }
.glow-safe { box-shadow: 0 0 20px rgba(6,214,160,0.15); }
.grid-bg {
  background-image: linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
  background-size: 40px 40px;
}
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
.animate-fade-in { animation: fadeIn 0.5s ease forwards; }
.animate-slide-up { animation: slideUp 0.4s ease forwards; }
""")

write("frontend/src/services/api.js", """import axios from 'axios'
const api = axios.create({ baseURL: '/api', timeout: 15000 })
export const predictToxicity = async (smiles, compoundName = '') => {
  const { data } = await api.post('/predict', { smiles, compound_name: compoundName || undefined })
  return data
}
export const predictBatch = async (compounds) => {
  const { data } = await api.post('/predict/batch', { compounds })
  return data
}
export const checkHealth = async () => {
  const { data } = await api.get('/health')
  return data
}
""")

write("frontend/src/components/SMILESInput.jsx", """import { useState } from 'react'
import { Search, Zap, ChevronDown, ChevronUp, FlaskConical } from 'lucide-react'

const EXAMPLES = [
  { name: "Aspirin", smiles: "CC(=O)Oc1ccccc1C(=O)O" },
  { name: "Ethanol", smiles: "CCO" },
  { name: "Nitrobenzene", smiles: "c1ccc(cc1)N(=O)=O" },
  { name: "Pyrene (PAH)", smiles: "c1cc2ccc3cccc4ccc(c1)c2c34" },
  { name: "Glycerol", smiles: "OCC(O)CO" },
  { name: "Carbon Tetrachloride", smiles: "ClC(Cl)(Cl)Cl" },
]

export default function SMILESInput({ onPredict, loading }) {
  const [smiles, setSmiles] = useState('')
  const [name, setName] = useState('')
  const [showExamples, setShowExamples] = useState(false)

  const handleSubmit = () => { if (!smiles.trim()) return; onPredict(smiles.trim(), name.trim()) }
  const handleExample = (ex) => { setSmiles(ex.smiles); setName(ex.name); setShowExamples(false) }

  return (
    <div className="w-full max-w-3xl mx-auto animate-fade-in">
      <div className="text-center mb-10">
        <div className="inline-flex items-center gap-2 bg-accent/10 border border-accent/20 text-accent text-xs font-mono px-3 py-1.5 rounded-full mb-4">
          <FlaskConical size={12} /> ML-POWERED · RDKIT · TOX21
        </div>
        <h1 className="text-4xl font-mono font-bold text-white mb-3 tracking-tight">
          Tox<span className="text-accent">Predict</span>
        </h1>
        <p className="text-muted text-sm max-w-md mx-auto">
          Predict drug toxicity from molecular structure using machine learning
        </p>
      </div>

      <div className="bg-card border border-border rounded-2xl p-6 glow-accent">
        <label className="block text-xs font-mono text-muted mb-2 uppercase tracking-widest">SMILES String</label>
        <textarea
          value={smiles} onChange={e => setSmiles(e.target.value)}
          placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O" rows={2}
          className="w-full bg-surface border border-border rounded-xl px-4 py-3 text-white font-mono text-sm placeholder:text-muted/40 focus:outline-none focus:border-accent/60 resize-none transition-all"
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSubmit())}
        />
        <label className="block text-xs font-mono text-muted mt-4 mb-2 uppercase tracking-widest">
          Compound Name <span className="text-muted/50">(optional)</span>
        </label>
        <input value={name} onChange={e => setName(e.target.value)} placeholder="e.g. Aspirin"
          className="w-full bg-surface border border-border rounded-xl px-4 py-3 text-white text-sm placeholder:text-muted/40 focus:outline-none focus:border-accent/60 transition-all" />

        <div className="flex items-center gap-3 mt-5">
          <button onClick={handleSubmit} disabled={!smiles.trim() || loading}
            className="flex-1 flex items-center justify-center gap-2 bg-accent text-bg font-mono font-bold py-3 rounded-xl hover:bg-accent/90 disabled:opacity-40 disabled:cursor-not-allowed transition-all text-sm">
            {loading ? <><div className="w-4 h-4 border-2 border-bg/40 border-t-bg rounded-full animate-spin" />ANALYZING...</> : <><Zap size={16} />PREDICT TOXICITY</>}
          </button>
          <button onClick={() => setShowExamples(v => !v)}
            className="flex items-center gap-2 border border-border text-muted hover:text-white hover:border-accent/40 font-mono text-xs px-4 py-3 rounded-xl transition-all">
            <Search size={14} />EXAMPLES{showExamples ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          </button>
        </div>

        {showExamples && (
          <div className="mt-4 grid grid-cols-2 gap-2 animate-fade-in">
            {EXAMPLES.map(ex => (
              <button key={ex.name} onClick={() => handleExample(ex)}
                className="text-left bg-surface border border-border hover:border-accent/40 rounded-xl px-4 py-3 transition-all group">
                <div className="text-white text-sm font-medium group-hover:text-accent transition-colors">{ex.name}</div>
                <div className="text-muted text-xs font-mono truncate mt-0.5">{ex.smiles}</div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
""")

write("frontend/src/components/ResultCard.jsx", """import { AlertTriangle, CheckCircle, XCircle, Info, Activity } from 'lucide-react'

const RISK_CONFIG = {
  HIGH:     { color: 'text-danger', bg: 'bg-danger/10', border: 'border-danger/30', glow: 'glow-danger', icon: XCircle,       label: 'HIGH RISK' },
  MODERATE: { color: 'text-warn',   bg: 'bg-warn/10',   border: 'border-warn/30',   glow: '',            icon: AlertTriangle,  label: 'MODERATE RISK' },
  LOW:      { color: 'text-yellow-400', bg: 'bg-yellow-400/10', border: 'border-yellow-400/30', glow: '', icon: AlertTriangle, label: 'LOW RISK' },
  MINIMAL:  { color: 'text-safe',   bg: 'bg-safe/10',   border: 'border-safe/30',   glow: 'glow-safe',   icon: CheckCircle,    label: 'MINIMAL RISK' },
}

function PropRow({ label, value, highlight }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-border/50 last:border-0">
      <span className="text-muted text-xs">{label}</span>
      <span className={\`text-xs font-mono font-bold \${highlight ? 'text-warn' : 'text-white'}\`}>{value}</span>
    </div>
  )
}

export default function ResultCard({ result }) {
  if (!result) return null
  if (!result.valid) return (
    <div className="w-full max-w-3xl mx-auto mt-6 animate-slide-up">
      <div className="bg-danger/10 border border-danger/30 rounded-2xl p-6 text-center">
        <XCircle className="text-danger mx-auto mb-3" size={32} />
        <p className="text-danger font-mono font-bold">INVALID SMILES</p>
        <p className="text-muted text-sm mt-1">{result.error}</p>
      </div>
    </div>
  )

  const config = RISK_CONFIG[result.risk_level] || RISK_CONFIG.MINIMAL
  const Icon = config.icon
  const props = result.molecular_properties || {}

  return (
    <div className="w-full max-w-3xl mx-auto mt-8 animate-slide-up space-y-4">
      <div className={\`\${config.bg} border \${config.border} \${config.glow} rounded-2xl p-6\`}>
        <div className="flex items-center justify-between mb-4">
          <div>
            {result.compound_name && <p className="text-muted text-xs font-mono uppercase tracking-widest mb-1">{result.compound_name}</p>}
            <div className="flex items-center gap-3">
              <Icon className={config.color} size={28} />
              <div>
                <p className={\`text-2xl font-mono font-bold \${config.color}\`}>{result.prediction}</p>
                <p className={\`text-xs font-mono \${config.color} opacity-70\`}>{config.label}</p>
              </div>
            </div>
          </div>
          <div className="text-right">
            <p className={\`text-4xl font-mono font-bold \${config.color}\`}>{result.toxic_probability}%</p>
            <p className="text-muted text-xs font-mono">TOXIC PROBABILITY</p>
          </div>
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs font-mono text-muted"><span>SAFE</span><span>TOXIC</span></div>
          <div className="flex gap-1 h-3 rounded-full overflow-hidden bg-surface">
            <div className="bg-safe transition-all duration-1000" style={{ width: \`\${result.safe_probability}%\` }} />
            <div className="bg-danger transition-all duration-1000" style={{ width: \`\${result.toxic_probability}%\` }} />
          </div>
          <div className="flex justify-between text-xs font-mono">
            <span className="text-safe">{result.safe_probability}%</span>
            <span className="text-danger">{result.toxic_probability}%</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-card border border-border rounded-2xl p-5">
          <div className="flex items-center gap-2 mb-4">
            <Activity size={14} className="text-accent" />
            <h3 className="text-xs font-mono text-accent uppercase tracking-widest">Molecular Properties</h3>
          </div>
          <PropRow label="Molecular Weight" value={\`\${props.molecular_weight} g/mol\`} highlight={props.molecular_weight > 500} />
          <PropRow label="LogP" value={props.logP} highlight={props.logP > 5} />
          <PropRow label="H-Bond Donors" value={props.h_bond_donors} highlight={props.h_bond_donors > 5} />
          <PropRow label="H-Bond Acceptors" value={props.h_bond_acceptors} highlight={props.h_bond_acceptors > 10} />
          <PropRow label="TPSA" value={\`\${props.tpsa} Å²\`} highlight={props.tpsa < 20} />
          <PropRow label="Aromatic Rings" value={props.aromatic_rings} highlight={props.aromatic_rings >= 3} />
          <PropRow label="Heavy Atoms" value={props.heavy_atom_count} />
          <div className={\`mt-4 text-center py-2 rounded-lg text-xs font-mono font-bold \${props.lipinski_violations <= 1 ? 'bg-safe/10 text-safe' : 'bg-warn/10 text-warn'}\`}>
            {props.drug_likeness} · {props.lipinski_violations} Lipinski violation{props.lipinski_violations !== 1 ? 's' : ''}
          </div>
        </div>

        <div className="bg-card border border-border rounded-2xl p-5">
          <div className="flex items-center gap-2 mb-4">
            <Info size={14} className="text-accent" />
            <h3 className="text-xs font-mono text-accent uppercase tracking-widest">Key Factors</h3>
          </div>
          <div className="space-y-3">
            {(result.toxicity_factors || []).map((factor, i) => (
              <div key={i} className="flex items-start gap-3 bg-surface rounded-xl px-3 py-2.5">
                <div className={\`w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0 \${config.color.replace('text-', 'bg-')}\`} />
                <p className="text-white text-xs leading-relaxed">{factor}</p>
              </div>
            ))}
          </div>
          <div className="mt-4 pt-4 border-t border-border">
            <p className="text-muted text-xs font-mono">MODEL</p>
            <p className="text-white text-xs mt-1">{result.model}</p>
          </div>
        </div>
      </div>

      <div className="bg-card border border-border rounded-xl px-5 py-3 flex items-center gap-3">
        <span className="text-muted text-xs font-mono uppercase tracking-widest flex-shrink-0">SMILES</span>
        <span className="text-accent text-xs font-mono truncate">{result.smiles}</span>
      </div>
    </div>
  )
}
""")

write("frontend/src/components/BatchAnalyzer.jsx", """import { useState } from 'react'
import { Layers, Plus, Trash2, Zap, Download } from 'lucide-react'
import { predictBatch } from '../services/api'

const RISK_COLORS = {
  HIGH: 'text-danger bg-danger/10 border-danger/30',
  MODERATE: 'text-warn bg-warn/10 border-warn/30',
  LOW: 'text-yellow-400 bg-yellow-400/10 border-yellow-400/30',
  MINIMAL: 'text-safe bg-safe/10 border-safe/30',
}

export default function BatchAnalyzer() {
  const [rows, setRows] = useState([
    { name: 'Aspirin', smiles: 'CC(=O)Oc1ccccc1C(=O)O' },
    { name: 'Nitrobenzene', smiles: 'c1ccc(cc1)N(=O)=O' },
    { name: 'Ethanol', smiles: 'CCO' },
  ])
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const addRow = () => setRows(r => [...r, { name: '', smiles: '' }])
  const removeRow = (i) => setRows(r => r.filter((_, idx) => idx !== i))
  const updateRow = (i, field, val) => setRows(r => r.map((row, idx) => idx === i ? { ...row, [field]: val } : row))

  const handleBatch = async () => {
    const valid = rows.filter(r => r.smiles.trim())
    if (!valid.length) return
    setLoading(true); setError(null)
    try {
      const data = await predictBatch(valid.map(r => ({ smiles: r.smiles.trim(), compound_name: r.name.trim() || undefined })))
      setResults(data)
    } catch (e) { setError('Backend not reachable. Make sure the server is running.') }
    finally { setLoading(false) }
  }

  const downloadCSV = () => {
    if (!results) return
    const headers = ['Compound','SMILES','Prediction','Toxic %','Risk Level','MW','LogP']
    const rows_data = results.results.map(r => [r.compound_name||'',r.smiles||'',r.prediction||'ERROR',r.toxic_probability||'',r.risk_level||'',r.molecular_properties?.molecular_weight||'',r.molecular_properties?.logP||''])
    const csv = [headers,...rows_data].map(r => r.join(',')).join('\\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url; a.download = 'toxicity_predictions.csv'; a.click()
  }

  return (
    <div className="w-full max-w-3xl mx-auto animate-fade-in">
      <div className="bg-card border border-border rounded-2xl p-6">
        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-2">
            <Layers size={16} className="text-accent" />
            <h2 className="text-sm font-mono text-accent uppercase tracking-widest">Batch Analysis</h2>
          </div>
          <button onClick={addRow} className="flex items-center gap-1.5 text-xs font-mono text-muted hover:text-accent border border-border hover:border-accent/40 px-3 py-1.5 rounded-lg transition-all">
            <Plus size={12} /> ADD ROW
          </button>
        </div>

        <div className="space-y-2 mb-5">
          <div className="grid grid-cols-5 gap-2 text-xs font-mono text-muted px-1 pb-1">
            <span className="col-span-1">NAME</span><span className="col-span-3">SMILES</span><span></span>
          </div>
          {rows.map((row, i) => (
            <div key={i} className="grid grid-cols-5 gap-2">
              <input value={row.name} onChange={e => updateRow(i,'name',e.target.value)} placeholder="Name"
                className="col-span-1 bg-surface border border-border rounded-lg px-3 py-2 text-xs text-white placeholder:text-muted/40 focus:outline-none focus:border-accent/60 transition-colors" />
              <input value={row.smiles} onChange={e => updateRow(i,'smiles',e.target.value)} placeholder="SMILES"
                className="col-span-3 bg-surface border border-border rounded-lg px-3 py-2 text-xs font-mono text-white placeholder:text-muted/40 focus:outline-none focus:border-accent/60 transition-colors" />
              <button onClick={() => removeRow(i)} disabled={rows.length===1}
                className="flex items-center justify-center text-muted hover:text-danger border border-border hover:border-danger/30 rounded-lg transition-all disabled:opacity-30">
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>

        <button onClick={handleBatch} disabled={loading || !rows.some(r => r.smiles.trim())}
          className="w-full flex items-center justify-center gap-2 bg-accent text-bg font-mono font-bold py-3 rounded-xl hover:bg-accent/90 disabled:opacity-40 disabled:cursor-not-allowed transition-all text-sm">
          {loading ? <><div className="w-4 h-4 border-2 border-bg/40 border-t-bg rounded-full animate-spin" />ANALYZING...</> : <><Zap size={16} />ANALYZE {rows.filter(r=>r.smiles.trim()).length} COMPOUNDS</>}
        </button>
        {error && <p className="text-danger text-xs font-mono text-center mt-3">{error}</p>}
      </div>

      {results && (
        <div className="mt-4 bg-card border border-border rounded-2xl p-6 animate-slide-up">
          <div className="grid grid-cols-3 gap-3 mb-5">
            {[{label:'TOTAL',val:results.total,color:'text-white'},{label:'TOXIC',val:results.toxic_count,color:'text-danger'},{label:'SAFE',val:results.safe_count,color:'text-safe'}].map(s=>(
              <div key={s.label} className="bg-surface rounded-xl p-3 text-center">
                <p className={\`text-2xl font-mono font-bold \${s.color}\`}>{s.val}</p>
                <p className="text-muted text-xs font-mono">{s.label}</p>
              </div>
            ))}
          </div>
          <div className="space-y-2">
            {results.results.map((r,i) => (
              <div key={i} className="flex items-center justify-between bg-surface rounded-xl px-4 py-3 gap-3">
                <div className="min-w-0 flex-1">
                  <p className="text-white text-sm font-medium truncate">{r.compound_name || \`Compound \${i+1}\`}</p>
                  <p className="text-muted text-xs font-mono truncate">{r.smiles}</p>
                </div>
                <div className="flex items-center gap-3 flex-shrink-0">
                  <span className="text-white text-sm font-mono">{r.toxic_probability}%</span>
                  <span className={\`text-xs font-mono font-bold px-2 py-1 rounded-lg border \${RISK_COLORS[r.risk_level]||'text-muted'}\`}>{r.risk_level||'N/A'}</span>
                </div>
              </div>
            ))}
          </div>
          <button onClick={downloadCSV} className="mt-4 w-full flex items-center justify-center gap-2 border border-border text-muted hover:text-accent hover:border-accent/40 font-mono text-xs py-2.5 rounded-xl transition-all">
            <Download size={14} /> DOWNLOAD CSV
          </button>
        </div>
      )}
    </div>
  )
}
""")

write("frontend/src/App.jsx", """import { useState, useEffect } from 'react'
import { FlaskConical, Layers, Wifi, WifiOff } from 'lucide-react'
import SMILESInput from './components/SMILESInput'
import ResultCard from './components/ResultCard'
import BatchAnalyzer from './components/BatchAnalyzer'
import { predictToxicity, checkHealth } from './services/api'

export default function App() {
  const [tab, setTab] = useState('single')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [health, setHealth] = useState(null)

  useEffect(() => {
    checkHealth().then(setHealth).catch(() => setHealth({ status: 'error' }))
  }, [])

  const handlePredict = async (smiles, name) => {
    setLoading(true); setError(null); setResult(null)
    try {
      const data = await predictToxicity(smiles, name)
      setResult(data)
    } catch (e) { setError('Cannot connect to backend. Make sure the server is running on port 8000.') }
    finally { setLoading(false) }
  }

  const serverOk = health?.status === 'ok'

  return (
    <div className="min-h-screen bg-bg grid-bg">
      <div className="border-b border-border bg-surface/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FlaskConical size={18} className="text-accent" />
            <span className="font-mono font-bold text-white text-sm">ToxPredict</span>
            <span className="text-muted text-xs font-mono hidden sm:block">/ Drug Toxicity ML</span>
          </div>
          <div className="flex items-center gap-4">
            {health && (
              <div className={\`flex items-center gap-1.5 text-xs font-mono \${serverOk ? 'text-safe' : 'text-danger'}\`}>
                {serverOk ? <Wifi size={12} /> : <WifiOff size={12} />}
                {serverOk ? (health.model_trained ? 'MODEL READY' : 'TRAIN MODEL') : 'SERVER DOWN'}
              </div>
            )}
            <div className="flex bg-card border border-border rounded-lg p-0.5">
              {[{id:'single',icon:FlaskConical,label:'Single'},{id:'batch',icon:Layers,label:'Batch'}].map(t=>(
                <button key={t.id} onClick={() => setTab(t.id)}
                  className={\`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-mono transition-all \${tab===t.id?'bg-accent text-bg font-bold':'text-muted hover:text-white'}\`}>
                  <t.icon size={12} />{t.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      <main className="max-w-4xl mx-auto px-6 py-12">
        {tab === 'single' ? (
          <>
            <SMILESInput onPredict={handlePredict} loading={loading} />
            {error && <div className="w-full max-w-3xl mx-auto mt-6 bg-danger/10 border border-danger/30 rounded-xl px-5 py-4"><p className="text-danger text-sm font-mono text-center">{error}</p></div>}
            <ResultCard result={result} />
          </>
        ) : <BatchAnalyzer />}

        {!health?.model_trained && health?.status === 'ok' && (
          <div className="w-full max-w-3xl mx-auto mt-8 bg-warn/10 border border-warn/30 rounded-xl px-5 py-4">
            <p className="text-warn text-xs font-mono text-center">
              Model not trained yet. Run: <span className="bg-surface px-2 py-0.5 rounded">cd backend/ml_model && python train_model.py</span>
            </p>
          </div>
        )}
      </main>

      <footer className="border-t border-border mt-16 py-6">
        <div className="max-w-4xl mx-auto px-6 flex items-center justify-between">
          <p className="text-muted text-xs font-mono">ToxPredict · CodeCure AI Hackathon 2026</p>
          <p className="text-muted text-xs font-mono">Random Forest · RDKit · Tox21</p>
        </div>
      </footer>
    </div>
  )
}
""")

write(".gitignore", """__pycache__/
*.py[cod]
venv/
.env
backend/ml_model/*.pkl
node_modules/
dist/
.DS_Store
""")

write("README.md", """# ToxPredict — Drug Toxicity Predictor
CodeCure AI Hackathon 2026

## Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
cd ml_model
python train_model.py
cd ..
uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173
""")

print("\n" + "=" * 40)
print("ALL FILES CREATED SUCCESSFULLY!")
print("=" * 40)
print("\nNext steps:")
print("  1. cd backend && pip install -r requirements.txt")
print("  2. cd ml_model && python train_model.py")
print("  3. cd .. && uvicorn app.main:app --reload")
print("  4. (new terminal) cd frontend && npm install && npm run dev")
print("\nOpen: http://localhost:5173")
