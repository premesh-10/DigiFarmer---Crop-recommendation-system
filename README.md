# 🌱 DigiFarmer — Crop Recommendation System

> **Intelligent crop recommendations through multimodal AI: computer vision + ensemble ML + economic heuristics**

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-F7931E?style=flat-square&logo=scikitlearn)](https://scikit-learn.org)
[![GitHub Stars](https://img.shields.io/github/stars/vlspranay/Crop-Recommendation-System?style=flat-square&logo=github)](https://github.com/vlspranay/Crop-Recommendation-System)

---

## 🎯 Overview

DigiFarmer is a three-tier AI decision-support system that transforms a single soil image into a multi-variate, economically-weighted crop recommendation. Unlike conventional ML-only pipelines, DigiFarmer couples ResNet50 deep vision inference with a deterministic **Composite Scoring Engine** that applies real-world agronomic, economic, and environmental constraints—preventing the "overconfident" predictions typical of unconstrained ML models.

---

## ✨ Key Features

- **🔬 Soil Image Classification** — Fine-tuned ResNet50 CNN classifies 8 soil types; peak validation accuracy **86.15%** over 50 training epochs
- **🧠 Ensemble Crop Recommendation** — Random Forest (100 estimators) trained on a 7-dimensional environmental vector (N, P, K, pH, Temperature, Humidity, Rainfall)
- **⚙️ Composite Scoring Engine** — Proprietary multi-factor formula fusing ML probability with soil suitability, profit margins, risk preference, and environmental constraints
- **🌐 Real-Time API Integration** — Live weather data & regional groundwater metrics fetched per request
- **🔄 Intelligent Data Imputation** — Missing environmental parameters are automatically filled using soil-type-specific defaults
- **🚀 High-Performance Backend** — Async FastAPI with both models globally pre-loaded (~500 MB RAM, <500ms tabular inference)
- **💻 Modern Glassmorphism UI** — Responsive HTML5/CSS3 frontend with drag-and-drop image upload

---

## 🏗️ Architecture

```
DigiFarmer/
├── 📄 run_app.py                       # Application entry point
├── 📄 requirements.txt                 # Root-level dependencies
├── 📄 soil_classificaton.ipynb         # ResNet50 training notebook
├── 📄 model_training.ipynb             # Random Forest training notebook
│
├── 📁 backend/
│   ├── 📄 requirements.txt             # Backend-specific dependencies
│   ├── 📁 api/
│   │   └── main.py                     # FastAPI routes & CORS setup
│   ├── 📁 models/
│   │   └── combined_crop_soil_recommender.py  # Core 3-tier inference engine
│   ├── 📁 data/
│   │   ├── crop_requirements_ap.json   # Crop environmental thresholds
│   │   ├── crop_economics.json         # Market price & cost margins
│   │   └── groundwater_ap.json         # Regional groundwater data
│   ├── 📁 services/                    # External API service modules
│   └── 📁 utils/                       # Shared utility functions
│
├── 📁 frontend/
│   ├── 📁 css/style.css                # Glassmorphism UI styling
│   ├── 📁 js/script.js                 # Interactive frontend logic
│   └── 📁 templates/index.html         # Single-page application
│
└── 📁 model_outputs/                   # ⚠️ Git-ignored — see below
    ├── soil_classifier_model.keras      # ResNet50 fine-tuned (95 MB)
    ├── water_stress_bilstm.keras        # BiLSTM stress model
    ├── crop_model.pkl                   # Random Forest classifier
    └── crop_label_encoder.pkl           # Label encoder for crops
```

> **⚠️ Note:** All files under `model_outputs/` are **git-ignored** (binary format, large sizes). See [Generating Model Files](#-generating-model-files) below.

---

## 🧠 AI Decision Pipeline

### Tier 1 — Computer Vision (ResNet50)
```
Soil Image → [224×224 Resize + Augmentation] → ResNet50 (ImageNet weights, top frozen)
           → GlobalAveragePooling → Dropout(0.2) → Dense(8, softmax)
           → Predicted Soil Class + Confidence
```
- **Training:** 50 epochs, Adam (lr=0.01), SparseCategoricalCrossentropy
- **Dataset:** 656 images, 8 classes, 90/10 train/validation split
- **Peak Validation Accuracy:** **86.15%** | Final Training Accuracy: **96.79%**

### Tier 2 — Tabular ML (Random Forest)
```
[N, P, K, pH, Temperature, Humidity, Rainfall] → Random Forest (100 estimators)
                                                → Raw Crop Probability P_RF
```
- **Data Imputation:** If tabular inputs are missing, the system automatically injects soil-type-specific optimal defaults, then fetches live weather and groundwater from external APIs.

### Tier 3 — Composite Scoring Engine
The raw ML probability is refined using a multi-modal fusion equation:

```
S_final = [ (0.35 × P_RF) + (0.20 × S_soil) + (0.20 × S_profit)
           + (0.15 × S_risk) + (0.10 × S_const) ] × W_factor
```

| Component | Weight | Source |
|---|:---:|---|
| Random Forest Probability `P_RF` | **35%** | Tier 2 ML output |
| Agronomic Soil Suitability `S_soil` | **20%** | CNN class → crop history mapping |
| Economic Profitability `S_profit` | **20%** | `(Price − Cost) / Price` from JSON |
| Risk Preference `S_risk` | **15%** | User risk appetite vs. crop volatility delta |
| Environmental Constraints `S_const` | **10%** | Temperature/rainfall biological tolerance bounds |
| Water Multiplier `W_factor` | ×1.05 / ×0.85 | Groundwater availability vs. crop demand |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- ~500 MB free RAM (for model loading)
- Model files present in `model_outputs/` (see below)

### 1. Clone the repository
```bash
git clone https://github.com/vlspranay/Crop-Recommendation-System.git
cd Crop-Recommendation-System
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r backend/requirements.txt
```

### 4. Generate model files *(first-time only)*
Model binaries are not included in this repo. Run the training notebooks to generate them:

```bash
# Opens Jupyter — run soil_classificaton.ipynb to produce soil_classifier_model.keras
jupyter notebook soil_classificaton.ipynb

# Run model_training.ipynb to produce crop_model.pkl + crop_label_encoder.pkl
jupyter notebook model_training.ipynb
```
Both notebooks output files to `model_outputs/` automatically.

### 5. Launch the application
```bash
python run_app.py
```

### 6. Access DigiFarmer
```
🌐 Web UI:    http://localhost:8000
📚 API Docs:  http://localhost:8000/api/docs
```

---

## 🔧 API Endpoints

| Endpoint | Method | Description | Latency |
|---|:---:|---|---|
| `/` | GET | Serves the main web interface | < 100ms |
| `/api/complete-analysis` | POST | Full pipeline: Image + tabular → recommendation | 2–5s |
| `/api/classify` | POST | ResNet50 soil classification only | 1–3s |
| `/api/recommend` | POST | Tabular-only crop recommendation (no image) | < 500ms |
| `/api/soil-types` | GET | Lists all 8 supported soil classes | < 100ms |
| `/api/health` | GET | System health + model load status | < 50ms |
| `/api/stats` | GET | Runtime statistics | < 100ms |

---

## 🌍 Supported Soil Types

| Soil Type | Best Suited Crops | pH Range |
|---|---|:---:|
| **Alluvial** | Rice, Wheat, Sugarcane, Cotton, Maize | 6.0–8.0 |
| **Black** | Cotton, Sugarcane, Wheat, Sunflower | 7.0–8.5 |
| **Cinder** | Coffee, Tea, Cardamom, Pepper | 5.5–7.0 |
| **Clay** | Rice, Wheat, Barley, Potatoes | 6.5–8.0 |
| **Laterite** | Cashew, Coconut, Rubber, Tea | 5.0–6.5 |
| **Peat** | Rice, Vegetables, Fruits, Herbs | 4.0–6.0 |
| **Red** | Groundnut, Potato, Rice, Pulses | 5.5–7.5 |
| **Yellow** | Wheat, Barley, Potato, Maize | 6.0–7.5 |

---

## 📊 Performance Metrics

| Metric | Value |
|---|---|
| ResNet50 Peak Validation Accuracy | **86.15%** (Epoch 2 & 5 & 21) |
| ResNet50 Final Training Accuracy | **96.79%** (Epoch 50) |
| Training Epochs | 50 |
| Full Multimodal Latency (Image + API) | 2–5 seconds |
| Tabular-Only Latency | < 500ms |
| Static Memory Footprint | ~500 MB RAM |
| Validation Set Size | 65 images (10% split) |

---

## 🔬 Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow / Keras, ResNet50 |
| Classical ML | Scikit-Learn (Random Forest) |
| Backend | FastAPI, Uvicorn, Pydantic |
| Data Processing | NumPy, Pandas, Pillow, Joblib |
| Frontend | HTML5, Vanilla JS, CSS3 (Glassmorphism) |
| External APIs | Weather & Groundwater APIs |

---

## 📁 Research Files (Included in Repo)

| File | Purpose |
|---|---|
| `evaluation_and_results.txt` | Full IEEE-style Evaluation & Results section |
| `research_paper_content.txt` | Complete research paper content extraction |
| `visuals_system_architecture.txt` | Mermaid system architecture diagram |
| `visuals_methodology_flowchart.txt` | Mermaid methodology flowchart |
| `visuals_confusion_matrix.txt` | 8×8 confusion matrix with analysis |
| `visuals_performance_graphs.txt` | Epoch data tables + Matplotlib plotting code |
| `visuals_composite_score.txt` | Composite score weight distribution chart |

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

<div align="center">

**🌱 Built with ❤️ for the future of agriculture**

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?style=for-the-badge&logo=python)](https://python.org)
[![Powered by TensorFlow](https://img.shields.io/badge/Powered%20by-TensorFlow-orange?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)

</div>
