# 🏥 Medical AI Diagnosis System v2.0

**Breast Cancer Diagnosis Prediction using Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/sklearn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic-yellow.svg)]()

---

## 🎯 Overview

Sistema de diagnóstico médico assistido por IA para classificação de tumores mamários (benignos/malignos) baseado em características citológicas. 

**⚠️ AVISO MÉDICO:** Esta é uma ferramenta de suporte à decisão clínica, **NÃO substitui** diagnóstico médico profissional.

### Key Features
- ✅ **97.37% accuracy** (Fase 1 baseline)
- ✅ **98.61% sensitivity** (detecta 71 de 72 casos)
- ✅ **REST API** com FastAPI
- ✅ **SHAP analysis** para interpretabilidade
- ✅ **Modular codebase** (refatorado da Fase 1)
- ✅ **Docker support** para deployment

---

## 🚀 Quick Start

### Option 1: Local Setup

```bash
# 1. Clone repository
cd Fase2/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model
python scripts/train_model.py --model logistic_regression

# 4. Start API
python scripts/run_api.py

# 5. Open browser
# http://localhost:8000/docs
```

### Option 2: Docker

```bash
# Build image
docker build -t medical-ai:v2.0 .

# Run container (after training model locally)
docker run -p 8000:8000 -v $(pwd)/models:/app/models medical-ai:v2.0

# Access API
curl http://localhost:8000/health
```

---

## 📁 Project Structure

```
Fase2/
├── README.md                    # This file
├── PLAN.md                      # Development plan
├── CONSTITUTION.md              # Project principles
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container definition
│
├── src/                         # SOURCE CODE (modular)
│   ├── data/                    # Data processing
│   │   ├── loader.py            # Load Wisconsin dataset
│   │   └── preprocessor.py     # Scaling + splitting
│   │
│   ├── models/                  # ML models
│   │   ├── trainer.py           # Train LR + RF
│   │   ├── evaluator.py         # Metrics + SHAP
│   │   └── predictor.py         # Inference
│   │
│   ├── api/                     # FastAPI
│   │   ├── main.py              # App entry point
│   │   ├── endpoints.py         # Routes
│   │   └── schemas.py           # Pydantic models
│   │
│   └── utils/                   # Utilities
│       ├── config.py            # Configuration
│       └── logger.py            # Logging
│
├── scripts/                     # SCRIPTS
│   ├── train_model.py           # Training pipeline
│   └── run_api.py               # API server
│
├── tests/                       # TESTS (pytest)
│   ├── test_preprocessor.py
│   ├── test_trainer.py
│   └── test_api.py
│
├── models/                      # Saved models (gitignored)
│   ├── logistic_regression.joblib
│   ├── scaler.joblib
│   └── metadata.json
│
└── data/                        # Data cache (gitignored)
```

---

## 🔬 Training Models

### Train Logistic Regression (default)

```bash
python scripts/train_model.py --model logistic_regression
```

**Output:**
- `models/logistic_regression.joblib` - Trained model
- `models/scaler.joblib` - StandardScaler
- `models/metadata.json` - Metrics and metadata

### Train Random Forest

```bash
python scripts/train_model.py --model random_forest
```

### Expected Results

| Metric | Target (Fase 1) | Tolerance |
|--------|-----------------|-----------|
| Accuracy | 97.37% | ± 1% |
| Sensitivity | 98.61% | ± 1% |
| Specificity | 95.24% | ± 1% |
| False Negatives | 1 | ≤ 2 |

---

## 🌐 API Usage

### Start Server

```bash
python scripts/run_api.py

# Custom host/port
python scripts/run_api.py --host 0.0.0.0 --port 8080
```

### Endpoints

#### `POST /predict` - Predict diagnosis

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "mean_radius": 17.99,
    "mean_texture": 10.38,
    "mean_perimeter": 122.8,
    "mean_area": 1001.0,
    "mean_smoothness": 0.1184,
    "mean_compactness": 0.2776,
    "mean_concavity": 0.3001,
    "mean_concave_points": 0.1471,
    "mean_symmetry": 0.2419,
    "mean_fractal_dimension": 0.07871,
    "radius_error": 1.095,
    "texture_error": 0.9053,
    "perimeter_error": 8.589,
    "area_error": 153.4,
    "smoothness_error": 0.006399,
    "compactness_error": 0.04904,
    "concavity_error": 0.05373,
    "concave_points_error": 0.01587,
    "symmetry_error": 0.03003,
    "fractal_dimension_error": 0.006193,
    "worst_radius": 25.38,
    "worst_texture": 17.33,
    "worst_perimeter": 184.6,
    "worst_area": 2019.0,
    "worst_smoothness": 0.1622,
    "worst_compactness": 0.6656,
    "worst_concavity": 0.7119,
    "worst_concave_points": 0.2654,
    "worst_symmetry": 0.4601,
    "worst_fractal_dimension": 0.1189
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "diagnosis": "Malignant",
  "probability": {
    "benign": 0.12,
    "malignant": 0.88
  },
  "confidence": 0.88
}
```

#### `GET /health` - Health check

```bash
curl http://localhost:8000/health
```

#### `GET /model-info` - Model information

```bash
curl http://localhost:8000/model-info
```

#### `GET /docs` - Interactive API documentation

Open in browser: http://localhost:8000/docs

---

## 🧪 Testing

### Run all tests

```bash
pytest tests/ -v
```

### With coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

### Test specific module

```bash
pytest tests/test_api.py -v
```

---

## 📊 Dataset

**Wisconsin Breast Cancer Dataset**
- **Source:** UCI ML Repository via sklearn.datasets
- **Samples:** 569 (357 benign, 212 malignant)
- **Features:** 30 cytological characteristics
- **Split:** 60% train / 20% val / 20% test (stratified)

### Feature Groups
1. **Mean features** (10): Average values
2. **Error features** (10): Standard error
3. **Worst features** (10): Mean of 3 largest values

---

## 🔑 Key Technical Decisions

### Why Logistic Regression?
✅ Equal performance to Random Forest (97.37%)  
✅ Better interpretability (linear coefficients)  
✅ Faster training and inference  
✅ More suitable for medical deployment  

### Why StandardScaler?
✅ Features have vastly different scales (0.05 to 4254)  
✅ Required for Logistic Regression convergence  
✅ Improves performance and stability  

### Why SHAP?
✅ Mandatory for medical AI interpretability  
✅ Regulatory requirement (FDA, ANVISA)  
✅ Builds trust with clinicians  
✅ Validates clinical relevance of features  

---

## 🏛️ Immutable Rules

From **CONSTITUTION.md**:

1. **random_state=42** - SEMPRE (reprodutibilidade total)
2. **Test set is sacred** - Nunca usar para tuning
3. **SHAP is mandatory** - Interpretabilidade obrigatória
4. **StandardScaler before ML** - Sempre escalar features
5. **Stratified splits** - Manter proporção das classes

---

## 📈 Comparison with Fase 1

| Aspect | Fase 1 | Fase 2 |
|--------|--------|--------|
| **Code** | Notebooks | Modular Python |
| **API** | None | FastAPI REST |
| **Tests** | Manual | Automated (pytest) |
| **Deployment** | Local only | Docker ready |
| **Structure** | Monolithic | Separated concerns |
| **Docs** | Basic | Comprehensive |

**Performance:** Maintained 97.37% accuracy from Fase 1 ✅

---

## 🚧 Known Limitations

1. **Single dataset:** Only trained on Wisconsin (no external validation)
2. **Small test set:** 114 samples (confidence intervals wide)
3. **1 False negative:** 1 cancer missed (critical)
4. **Tabular only:** No image data (no CNN)
5. **Support tool:** Not FDA/ANVISA approved

---

## 🔮 Future Roadmap

### Fase 3 (Production)
- [ ] Web interface (React/Vue)
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Advanced logging (ELK stack)
- [ ] Monitoring (Prometheus)
- [ ] External dataset validation

### Fase 4 (Clinical)
- [ ] Multi-hospital trials
- [ ] Regulatory approval
- [ ] Integration with PACS
- [ ] Real-time monitoring

---

## 🤝 Contributing

This is an academic project (FIAP Tech Challenge). Contributions follow:

1. Read **CONSTITUTION.md** (principles)
2. Follow code style (black, flake8)
3. Write tests (pytest)
4. Update docs (README, docstrings)

---

## 📄 License

**Academic Use Only** - FIAP Tech Challenge Fase 2

---

## 📞 Support

### Documentation
- **PLAN.md** - Development plan
- **CONSTITUTION.md** - Project principles
- **API Docs** - http://localhost:8000/docs

### Issues
For questions about:
- **Setup:** Check this README
- **API:** Check /docs endpoint
- **Code:** Check docstrings in source

---

## 🎓 Citation

If using this project, cite:

```
Medical AI Diagnosis System v2.0
FIAP - Tech Challenge Fase 2
2026
```

---

## ✅ Quick Validation

After setup, verify everything works:

```bash
# 1. Train model
python scripts/train_model.py

# 2. Run tests
pytest tests/ -v

# 3. Start API
python scripts/run_api.py &

# 4. Test endpoint
curl http://localhost:8000/health

# 5. Check docs
open http://localhost:8000/docs
```

Expected: All commands succeed, API responds, tests pass.

---

**Version:** 2.0.0  
**Last Updated:** 2026-01-07  
**Status:** ✅ Production Ready (Academic)  

**⚠️ Medical Disclaimer:** This is a research/educational tool. Always consult qualified medical professionals for diagnosis and treatment.
