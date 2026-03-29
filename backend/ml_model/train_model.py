"""
train_model.py
==============
ML model training script for drug toxicity prediction.

How it works:
    1. Define known toxic and safe SMILES compounds
    2. Extract Morgan fingerprints + RDKit descriptors from each
    3. Train a Random Forest classifier on these features
    4. Evaluate model accuracy and print classification report
    5. Save trained model as toxicity_model.pkl

Run this once before starting the backend:
    cd backend/ml_model
    python train_model.py

Output files:
    toxicity_model.pkl — trained Random Forest model
    model_info.pkl     — metadata (feature count, etc.)
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Try to import RDKit for molecular feature extraction
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ── Training Data ──────────────────────────────────────────────
# Curated list of known TOXIC compounds from Tox21 dataset
# Includes PAHs, halogenated compounds, nitro compounds, etc.
TOXIC_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",        # Aspirin (GI toxicity)
    "c1ccc2ccccc2c1",                # Naphthalene (PAH)
    "C1=CC=C(C=C1)N",               # Aniline (hepatotoxic)
    "ClC1=CC=CC=C1",                # Chlorobenzene
    "BrC1=CC=CC=C1",                # Bromobenzene
    "Clc1ccc(Cl)cc1",               # 1,4-Dichlorobenzene
    "c1ccc(cc1)N(=O)=O",            # Nitrobenzene
    "C(Cl)(Cl)(Cl)Cl",             # Carbon tetrachloride (hepatotoxic)
    "ClCC(Cl)Cl",                   # Trichloroethane-like
    "c1ccncc1",                     # Pyridine
    "c1ccc(cc1)Cl",                 # Chlorobenzene
    "N#Cc1ccccc1",                  # Benzonitrile
    "O=C1NC(=O)NC1=O",             # Barbituric acid
    "c1ccc2[nH]ccc2c1",            # Indole
    "O=C(O)c1ccccc1O",             # Salicylic acid
    "O=C(c1ccccc1)c1ccccc1",       # Benzophenone
    "c1ccc(-c2ccccc2)cc1",          # Biphenyl
    "c1ccc2c(c1)ccc1ccccc12",      # Anthracene (PAH)
    "c1cc2ccc3cccc4ccc(c1)c2c34",  # Pyrene (PAH - highly toxic)
    "ClCCl",                         # Dichloromethane
    "BrCBr",                         # Dibromomethane
    "O=C1OCC1",                      # Beta-propiolactone (carcinogen)
    "C1CO1",                         # Ethylene oxide (carcinogen)
    "CC(C)(C)c1ccc(cc1)N",          # 4-tert-butylaniline
    "CC(=O)c1ccc(cc1)O",            # 4-Hydroxyacetophenone
]

# Known SAFE compounds — low toxicity solvents, nutrients, amino acids
SAFE_SMILES = [
    "CCO",                            # Ethanol
    "CC(C)O",                         # Isopropanol
    "OCC(O)CO",                       # Glycerol
    "OC1CCCCC1",                      # Cyclohexanol
    "CC(=O)O",                        # Acetic acid
    "OC(=O)C(O)=O",                   # Oxalic acid
    "OC(CO)(CO)CO",                   # Pentaerythritol
    "OCC(O)C(O)C(O)C(O)CO",          # Sorbitol
    "CC1=CC=CC=C1",                   # Toluene
    "CCCO",                           # 1-Propanol
    "CCCCO",                          # 1-Butanol
    "C1CCCCC1",                       # Cyclohexane
    "CCCCCC",                         # Hexane
    "C1CCCC1",                        # Cyclopentane
    "CC(C)=O",                        # Acetone
    "CCOC(C)=O",                      # Ethyl acetate
    "COC(=O)C",                       # Methyl acetate
    "CCOCC",                          # Diethyl ether
    "C1CCCOC1",                       # Tetrahydrofuran
    "NCC(=O)O",                       # Glycine (amino acid)
    "CC(N)C(=O)O",                    # Alanine (amino acid)
    "OC(Cc1ccccc1)C(=O)O",           # Phenylalanine (amino acid)
    "CC(C)(N)C(=O)O",                 # Valine (amino acid)
    "OCC",                            # Ethylene glycol
]


def get_features(smiles: str):
    """
    Extract a numerical feature vector from a SMILES string.

    Combines two types of features:
        1. Morgan Fingerprint (2048 bits) — encodes molecular topology
           using circular algorithm with radius=2 (same as ECFP4)
        2. 20 RDKit physicochemical descriptors — MW, LogP, TPSA, etc.

    Returns:
        list of floats (length 2068), or None if SMILES is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Invalid SMILES — skip this compound

    # Morgan fingerprint: 2048-bit binary vector
    fp = list(GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))

    # Physicochemical descriptors
    desc = [
        Descriptors.MolWt(mol),                                          # Molecular weight
        Descriptors.MolLogP(mol),                                        # Lipophilicity
        Descriptors.NumHDonors(mol),                                     # H-bond donors
        Descriptors.NumHAcceptors(mol),                                  # H-bond acceptors
        Descriptors.TPSA(mol),                                           # Polar surface area
        Descriptors.NumRotatableBonds(mol),                              # Molecular flexibility
        rdMolDescriptors.CalcNumAromaticRings(mol),                      # Aromatic ring count
        mol.GetNumHeavyAtoms(),                                          # Heavy atom count
        rdMolDescriptors.CalcNumRings(mol),                              # Total ring count
        Descriptors.FractionCSP3(mol),                                   # sp3 carbon fraction
        Descriptors.BalabanJ(mol) if mol.GetNumBonds() > 0 else 0,      # Balaban index
        Descriptors.BertzCT(mol),                                        # Molecular complexity
        Descriptors.Chi0(mol),                                           # Chi path index
        Descriptors.Chi1(mol),                                           # Chi path index
        Descriptors.HallKierAlpha(mol),                                  # Hall-Kier alpha
        Descriptors.Kappa1(mol),                                         # Kappa shape index
        Descriptors.Kappa2(mol),                                         # Kappa shape index
        Descriptors.LabuteASA(mol),                                      # Surface area
        Descriptors.MaxPartialCharge(mol),                               # Max atomic charge
        Descriptors.MinPartialCharge(mol),                               # Min atomic charge
    ]

    # Concatenate fingerprint bits + descriptor values
    return fp + desc


# ── Main Training Script ───────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Drug Toxicity ML Model - Training")
    print("=" * 50)

    # Exit early if RDKit is not available
    if not RDKIT_AVAILABLE:
        print("ERROR: RDKit not installed. Run: pip install rdkit")
        exit(1)

    # ── Build Dataset ──────────────────────────────────────────
    print("\nExtracting molecular features...")
    X, y = [], []

    # Label 1 = TOXIC
    for smi in TOXIC_SMILES:
        f = get_features(smi)
        if f is not None:
            X.append(f)
            y.append(1)

    # Label 0 = SAFE
    for smi in SAFE_SMILES:
        f = get_features(smi)
        if f is not None:
            X.append(f)
            y.append(0)

    X = np.array(X)
    y = np.array(y)
    print(f"Dataset: {len(X)} compounds ({sum(y)} toxic, {len(y)-sum(y)} safe)")

    # ── Train/Test Split ───────────────────────────────────────
    # 80% training, 20% testing
    # stratify=y ensures both splits have equal toxic/safe ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ── Train Model ────────────────────────────────────────────
    # Random Forest with 200 trees
    # class_weight="balanced" handles class imbalance automatically
    # n_jobs=-1 uses all available CPU cores for speed
    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────
    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Toxic"]))

    # ── Save Model ─────────────────────────────────────────────
    # Saved in the same directory as this script (ml_model/)
    with open("toxicity_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("model_info.pkl", "wb") as f:
        pickle.dump({"n_features": X.shape[1]}, f)

    print("\nModel saved: toxicity_model.pkl")
    print("Training complete! You can now start the backend.")