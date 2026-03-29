import pickle
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

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "ml_model", "toxicity_model.pkl")
INFO_PATH  = os.path.join(BASE_DIR, "ml_model", "model_info.pkl")
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
