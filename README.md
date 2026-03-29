# 🧪 ToxPredict — Drug Toxicity Predictor

> AI-powered drug toxicity prediction using Machine Learning and molecular descriptors

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=flat-square&logo=fastapi)
![React](https://img.shields.io/badge/React-18.3-61DAFB?style=flat-square&logo=react)
![RDKit](https://img.shields.io/badge/RDKit-2025-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

**CodeCure AI Hackathon 2026 · Track A: Drug Toxicity Prediction (Pharmacology + AI)**

---

## 🎯 Problem Statement

Drug development frequently fails due to unexpected toxicity. Early prediction of toxic compounds can significantly reduce development costs and improve patient safety.

ToxPredict uses **Machine Learning** to predict potential drug toxicity from chemical structure, helping researchers identify risky compounds before expensive lab testing.

---

## ✨ Features

- 🔬 **Single Compound Analysis** — Enter any SMILES string and get instant toxicity prediction
- ⚗️ **2D Molecular Structure** — Visual diagram of the molecule rendered using RDKit
- 📊 **Molecular Properties** — MW, LogP, TPSA, H-bond donors/acceptors, Lipinski Rule of 5
- ⚠️ **Risk Assessment** — 4-level risk scoring (MINIMAL / LOW / MODERATE / HIGH)
- 📦 **Batch Analysis** — Analyze multiple compounds at once
- 📥 **CSV Export** — Download batch results as spreadsheet
- 🧠 **ML Model** — Random Forest trained on Morgan Fingerprints + RDKit Descriptors

---

## 🏗️ Architecture
```
drug-toxicity-predictor/
├── backend/                        # Python FastAPI server
│   ├── app/
│   │   ├── main.py                 # App entry point + CORS setup
│   │   ├── api/
│   │   │   ├── predict.py          # /predict, /batch, /molecule-image
│   │   │   └── health.py           # /health status endpoint
│   │   ├── models/
│   │   │   └── schemas.py          # Pydantic request/response models
│   │   └── services/
│   │       └── predict_service.py  # ML model + feature extraction
│   ├── ml_model/
│   │   └── train_model.py          # Model training script
│   └── requirements.txt
└── frontend/                       # React + Vite + Tailwind
    └── src/
        ├── App.jsx                 # Root component + tab navigation
        ├── components/
        │   ├── SMILESInput.jsx     # Molecule input form
        │   ├── ResultCard.jsx      # Prediction result display
        │   └── BatchAnalyzer.jsx   # Batch analysis tool
        └── services/
            └── api.js              # HTTP calls to backend
```

---

## 🤖 ML Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Classifier |
| Fingerprint | Morgan (ECFP4) — 2048 bits |
| Descriptors | 20 RDKit physicochemical descriptors |
| Total Features | 2068 per molecule |
| Training Data | Tox21-inspired toxic/safe compound dataset |
| Class Balancing | Balanced class weights |

### Molecular Descriptors Used
- Molecular Weight, LogP, TPSA
- H-Bond Donors & Acceptors
- Aromatic Rings, Rotatable Bonds
- Kappa Shape Indices, Chi Path Indices
- Balaban J Index, Bertz Complexity
- Partial Charges (Max & Min)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+

### Backend Setup
```bash
cd backend

# Install dependencies
pip install fastapi uvicorn pydantic scikit-learn numpy pandas rdkit python-multipart

# Train the ML model
cd ml_model
python train_model.py

# Start the server
cd ..
python -m uvicorn app.main:app --reload
```

Backend running at: `http://localhost:8000`
API docs at: `http://localhost:8000/docs`

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Frontend running at: `http://localhost:5173`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/predict` | Predict toxicity of single compound |
| `POST` | `/api/predict/batch` | Predict toxicity for multiple compounds |
| `GET` | `/api/molecule-image/{smiles}` | Get 2D structure image as PNG |
| `GET` | `/api/examples` | Get example SMILES for testing |
| `GET` | `/api/health` | Server + model status |

### Example Request
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "c1ccc(cc1)N(=O)=O", "compound_name": "Nitrobenzene"}'
```

---

## 🧪 Example Compounds

| Compound | SMILES | Risk Level |
|----------|--------|------------|
| Ethanol | `CCO` | 🟢 MINIMAL |
| Aspirin | `CC(=O)Oc1ccccc1C(=O)O` | 🟡 LOW |
| Naphthalene | `c1ccc2ccccc2c1` | 🟠 MODERATE |
| Nitrobenzene | `c1ccc(cc1)N(=O)=O` | 🔴 HIGH |
| Pyrene (PAH) | `c1cc2ccc3cccc4ccc(c1)c2c34` | 🔴 HIGH |
| Carbon Tetrachloride | `ClC(Cl)(Cl)Cl` | 🔴 HIGH |

---

## 📋 Hackathon Deliverables

- [x] GitHub Repository with clean code and comments
- [x] Machine learning model predicting drug toxicity
- [x] Feature importance via structural alerts
- [x] 2D molecular structure visualization
- [x] Toxicity probability with risk level scoring
- [x] Simple prediction interface (web app)
- [x] Batch analysis tool with CSV export
- [x] Lipinski Rule of 5 drug-likeness assessment

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite, Tailwind CSS |
| Backend | Python, FastAPI, Pydantic |
| ML Model | scikit-learn, Random Forest |
| Cheminformatics | RDKit, Morgan Fingerprints |
| Dataset | Tox21-inspired compound library |

---

## 👥 Team

Built for **CodeCure AI Hackathon 2026** · Track A: Drug Toxicity Prediction