# ToxPredict — Drug Toxicity Predictor
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
