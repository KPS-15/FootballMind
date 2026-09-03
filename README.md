# FootballMind: Multimodal Deep Learning & Spatial Physics Framework for Football Intelligence

**FootballMind** is an end-to-end multimodal deep learning, kinematic physics, and spatial intelligence framework engineered for predictive, tactical, and explainable football match analytics. It unifies raw tracking data and broadcast video ingestion with multi-class action intention forecasting, Time-To-Intercept (TTI) pass recommendation ranking, direction-aware defensive collapse structural scoring, calibrated logistic expected goals (xG), counterfactual What-If tactical simulations, empirical feature occlusion attribution (XAI), and automated computer vision video overlay generation.

---

## ⚽ Key Framework Capabilities

1. **Perception & MOT Tracking**: Object detection via YOLOv8/v11 (`player`, `referee`, `goalkeeper`, `ball`), multi-object tracking data association via the **Hungarian Algorithm** (`scipy.optimize.linear_sum_assignment`) over Euclidean distance matrices, player pose/orientation estimation, and persistent track-ID management.
2. **SigLIP & HSV Jersey Team Identification**: Upper-torso cropping with HSV green grass background filtering ($35 \le H \le 85$), Hugging Face SigLIP visual embeddings (`google/siglip-base-patch16-224`) combined with multi-channel color features, Scikit-Learn K-Means clustering ($k=2$), and track-ID temporal majority voting.
3. **Homography & Top-Down Metric Mapping**: OpenCV RANSAC perspective transformation mapping pixel coordinates $(x_{\text{pixel}}, y_{\text{pixel}})$ to standardized top-down $105\text{m} \times 68\text{m}$ pitch coordinates.
4. **16D Unified Football State Encoder**: Standardized spatial feature vector per player encoding 2D velocities, acceleration, body orientation, nearest teammate/opponent distances, local density (8m radius), open pitch space, goal distance, and ball proximity.
5. **PyTorch Temporal Action Intention Predictor**: 2-layer LSTM temporal sequence model trained on 8 action classes (`PASS`, `DRIBBLE`, `SHOT`, `CROSS`, `TACKLE`, `CARRY`, `HOLD`, `CLEARANCE`), equipped with a temperature-scaled ($\tau = 1.2$) multinomial softmax fallback baseline.
6. **Time-To-Intercept (TTI) Kinematic Pass Recommender**: Ranks pass options using exponential ball velocity drag decay $v(t) = v_0 e^{-\mu t}$, calculating ball arrival times vs. opponent sprint reaction arrival times.
7. **Direction-Aware Defensive Collapse Index**: Continuous structural danger scoring measuring CB-LB lateral gap, CB-RB lateral gap, midfield exposure, Zone 14 passing lane vulnerability, and unmarked attacker space.
8. **Calibrated Logistic Sigmoid xG Model**: Evaluates goal probability $\sigma(\boldsymbol{\beta}^T \mathbf{x})$ based on shot distance, subtended goal post angle, and defending player density inside the shooting cone.
9. **Counterfactual What-If Tactical Simulator**: Real-time positional delta engine recalculating exact changes in defensive danger ($\Delta \text{Danger}$), expected goals ($\Delta \text{xG}$), and spatial control ($\Delta \text{Space}$) when player coordinates are adjusted.
10. **Empirical Model Feature Occlusion (XAI)**: Quantifies model feature attributions by systematically perturbing/masking input dimensions and measuring target action probability drops $\Delta P(\text{target})$, paired with 5–10s goal sequence failure reconstruction.
11. **End-to-End Video Analysis & Tactical Overlay Pipeline**: Automated video upload (MP4/MOV/AVI/MKV up to 500MB), computer vision tracking, dynamic tactical HUD generation, player team badges, animated pass vectors, defensive mark lines, and downloadable annotated MP4 export.
12. **Scientific Evaluation & Experiment Tracking**: Automated calculation of Expected Calibration Error (ECE) via 10-bin reliability diagrams, Brier score, Log loss, Macro F1-score, $8 \times 8$ confusion matrix, and persistent JSON/CSV experiment logging.

---

## 🏗️ System Architecture

```
+-----------------------------------------------------------------------------------+
|                            1. PERCEPTION TIER (Vision)                            |
|  Ultralytics YOLOv8/v11  -->  Hungarian MOT Tracker  -->  SigLIP/HSV Clustering   |
|                                         |                                         |
|                           cv2.findHomography RANSAC Matrix                        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        2. STATE ENCODING TIER (Core)                              |
|   Translates pixel space (1920x1080) to pitch metric coordinates (105m x 68m)     |
|   Extracts 16-Dimensional Normalized Spatial Feature Vector per player per frame  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                       3. INTELLIGENCE & PHYSICS ENGINE                            |
|  - Action Predictor: PyTorch Temporal LSTM & Calibrated Softmax Baseline          |
|  - Pass Predictor: Time-To-Intercept (TTI) Kinematic Drag Model                   |
|  - Defensive Analyzer: Direction-Aware Spatial Geometry Index                     |
|  - Goalkeeper Analyzer: Calibrated Logistic Sigmoid xG Model                      |
|  - What-If Simulator: Counterfactual Positional Delta Evaluator                   |
|  - Explainability Engine: Empirical Model Feature Occlusion Perturbation          |
|  - Video Analyzer: Automated Tactical Sequence Classification & Event Timeline    |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                      4. PRESENTATION & API TIER (FastAPI/Next.js)                 |
|  FastAPI REST Server (8000)  -->  Next.js 14 Dual-Mode Tactical Dashboard (3000)   |
|   - Interactive Tactical View (Draggable Canvas Pitch & Real-Time Intelligence)   |
|   - Video Upload & Analysis View (Annotated Video Player, Overlays, & Downloads)  |
+-----------------------------------------------------------------------------------+
```

---

## 📊 Dashboard & UI Interfaces

The frontend is built with **Next.js 14**, **TypeScript**, and **Tailwind CSS**, featuring two dedicated operational modes:

### 1. Interactive Tactical View
- **Interactive SVG Pitch**: Real-time rendering of all 22 players and the ball with team color coding, jersey numbers, velocity vectors, and pass recommendation lines.
- **Draggable What-If Simulator**: Drag any player token across the pitch to trigger real-time counterfactual simulation recalculating defensive risk deltas and xG variations.
- **Playback & Scrubber**: Frame-by-frame temporal timeline controller with Play, Pause, and Reset controls.
- **Prediction & Tactical Intelligence Cards**: Real-time action intention probabilities, top alternative decisions, TTI pass option leaderboards, defensive collapse breakdown gauges, and goalkeeper positioning recommendations.
- **Explainability & Goal Breakdown**: Empirical feature attribution charts and goal sequence failure breakdown.

### 2. Video Upload & Analysis Hub
- **Direct Video Upload**: Upload match clips up to 500 MB in MP4, MOV, AVI, or MKV format.
- **4-Stage CV Processing Stepper**: Real-time tracking of YOLO object detection, Hungarian MOT tracking, ball tracking, and tactical calculation.
- **Dual-Mode Video Player**: Seamless toggle between **Analyzed** (with animated pass arrows, marking lines, player badges, HUD, and ball tracking) and **Original** footage.
- **Downloadable MP4 Export**: Direct download of annotated match video clips with embedded tactical overlays.
- **Live Sports Analytics Cards**: Instant tactical readout in high-contrast broadcast cards:
  - **Attack Card**: Ball carrier identification, best action recommendation, tactical score, and space reasoning.
  - **Defence Card**: Defensive danger score (0–100), primary attacking threat, and actionable defensive assignments (`MARK`, `PRESS`, `COVER`).
  - **Tactical Alert**: Instant situational warning and recommended team response (`SHIFT LEFT` / `SHIFT RIGHT`).
  - **Key Timeline**: Timestamped sequence of key tactical match moments.
  - **Detailed Report Log**: Expandable match summary log for deep technical review.

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- **Python**: 3.10 or higher
- **Node.js**: v18 or higher (with `npm`)

### 2. Clone & Environment Setup
```bash
# Clone the repository
git clone https://github.com/footballmind/footballmind.git
cd footballmind

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate
# Activate virtual environment (Linux/macOS)
# source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. YOLO11m Vision Perception & Training Pipeline

FootballMind uses **Ultralytics YOLO11m** as the default object detection model for player, referee, goalkeeper, and small-object football detection.

#### Dataset Preparation (Roboflow / Ultralytics)
Export your dataset from Roboflow in **YOLOv11** or **YOLOv8 PyTorch** format. Ensure `datasets/data.yaml` defines `train`, `val`, and `names`:
```yaml
path: datasets/
train: train/images
val: val/images
test: test/images # optional
names:
  0: ball
  1: goalkeeper
  2: player
  3: referee
```

#### Training YOLO11m Detector
```bash
# 1. Validate dataset without training (Dry Run)
python -m training.train_detector --dry-run --data datasets/data.yaml

# 2. Train YOLO11m on Roboflow dataset (Default: model=yolo11m.pt, imgsz=1280, epochs=100)
python -m training.train_detector --data datasets/data.yaml --model yolo11m.pt --imgsz 1280 --epochs 100 --batch 16

# Or using the FootballMind CLI
footballmind train-detector --data datasets/data.yaml --model yolo11m.pt
```
*Note: Hardware auto-scaling automatically reduces batch size (e.g. 4) and resolution if GPU memory is constrained or on CPU.*

#### Evaluating Detector & Latency Benchmarks
```bash
# Evaluate on dataset (Precision, Recall, mAP50, mAP50-95, Ball Recall, Player Recall)
python -m training.evaluate_detector --data datasets/data.yaml --model yolo11m.pt

# Run Latency & Throughput Benchmark
python -m training.evaluate_detector --benchmark-only --model yolo11m.pt

# Train PyTorch Temporal LSTM Action Predictor (15 epochs)
python -m training.train_action

# Run Full Model & Vision Evaluation Suite
python -m training.evaluate
```

#### Image & Video Inference Commands
```bash
# Run detection on a single image file (Local YOLO11m)
footballmind detect-image path/to/frame.jpg --model yolo11m.pt --conf 0.35 --ball-conf 0.20

# Run Roboflow Hosted Serverless Workflow (general-segmentation-api) on an image
footballmind roboflow-workflow path/to/frame.jpg

# Process and annotate video clip with tactical overlays
footballmind process-video path/to/match.mp4 --output outputs/annotated_match.mp4 --max-frames 200
```

#### Roboflow Hosted Serverless Workflow Integration
You can use Roboflow's Serverless Workflow API (`inference-sdk`) as an enhanced cloud segmentation backend:
```python
from src.vision.roboflow_client import RoboflowWorkflowClient

client = RoboflowWorkflowClient(
    api_url="https://serverless.roboflow.com",
    api_key="6iE3b3FVMQzewTLHfWDT",
    workspace_name="k-p-shohil",
    workflow_id="general-segmentation-api"
)

# Run workflow on an image file or numpy frame
result = client.run_workflow_on_image("path/to/frame.jpg")

# Directly extract normalized FootballMind DetectedObject structures
detections = client.detect_frame(frame)
```

### 4. Launch FastAPI Backend & Next.js Tactical Dashboard

**Terminal 1 — Backend Server**:
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
*API Swagger Documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).*

**Terminal 2 — Frontend Dashboard**:
```bash
cd frontend
npm install
npm run dev
```
*Open [http://localhost:3000](http://localhost:3000) in your browser.*

---

## 🧪 Automated Testing

FootballMind includes a comprehensive test suite covering core state encoders, vision and tracking pipelines, homography, tactical analyzers, ML models, What-If simulation, explainability, video analysis, and REST API endpoints.

Execute the test suite with Pytest:
```bash
python -m pytest
```
```text
tests/test_api.py .........                                              [ 37%]
tests/test_core.py ....                                                  [ 54%]
tests/test_evaluation.py ..                                              [ 62%]
tests/test_explainability.py ..                                          [ 70%]
tests/test_simulation.py .                                               [ 75%]
tests/test_tactical.py ...                                               [ 87%]
tests/test_video_analysis.py ...                                         [100%]

======================= 24 passed in 220.61s =======================
```

---

## 📡 REST API Reference

The backend provides a fully typed FastAPI REST API:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | System health check & API version status |
| `GET` | `/api/match/{match_id}` | Metadata and settings for the specified match |
| `GET` | `/api/tracking/{match_id}` | Frame-level tracking state (22 players + ball coordinates) |
| `GET` | `/api/prediction/{match_id}` | Action prediction and receiver probability for a player |
| `GET` | `/api/tactical/{match_id}` | Defensive collapse index, xG, and GK recommendation |
| `GET` | `/api/recommendations/{match_id}` | Ranked TTI kinematic pass options for the ball carrier |
| `POST` | `/api/simulation` | Counterfactual What-If tactical positional delta evaluation |
| `GET` | `/api/explanation/{match_id}` | Empirical feature attributions & goal sequence reconstruction |
| `GET` | `/api/evaluation/{match_id}` | ECE, Brier score, Log loss, Macro F1, and confusion matrix |
| `POST` | `/api/video/upload` | Upload video file (MP4/MOV/AVI/MKV) for CV processing |
| `POST` | `/api/video/analyze` | Run full CV detection, MOT tracking, and tactical video analysis |
| `GET` | `/api/video/download/{filename}` | Download generated annotated MP4 video file |
| `POST` | `/api/video/process` | Raw tracking video processing endpoint (backward compatible) |

*Full request/response schema specifications are documented in [docs/API.md](docs/API.md).*

---

## 📁 Project Directory Structure

```
footballmind/
├── backend/                  # FastAPI REST API implementation
│   ├── routes/               # API route handlers
│   │   ├── match.py          # Match, tracking, prediction, tactical, simulation, evaluation
│   │   ├── simulation.py     # Standalone simulation router
│   │   └── video.py          # Video upload, analysis, and download routes
│   ├── main.py               # FastAPI application setup, CORS, static mounts
│   └── __init__.py
├── frontend/                 # Next.js 14 TypeScript web dashboard
│   ├── src/
│   │   ├── app/              # Next.js App Router (layout.tsx, page.tsx, globals.css)
│   │   ├── components/       # Dashboard UI components
│   │   │   ├── BestPassCard.tsx         # TTI pass recommendation ranking
│   │   │   ├── DefensiveDangerCard.tsx  # Defensive collapse index & gauges
│   │   │   ├── ExplainabilityCard.tsx   # Feature occlusion & goal sequence XAI
│   │   │   ├── PredictionCard.tsx       # Action intention forecasting card
│   │   │   ├── TacticalPitch.tsx        # Interactive SVG pitch with draggable nodes
│   │   │   ├── VideoUploadSection.tsx   # Video upload, processing & hero player
│   │   │   └── WhatIfSimulatorCard.tsx  # Counterfactual What-If control card
│   │   └── types/            # TypeScript type definitions (football.ts)
│   ├── package.json
│   └── tailwind.config.ts
├── src/                      # Core Intelligence Framework
│   ├── core/                 # State Encoder (16D), Config, Data Types & Models
│   │   ├── config.py         # System configuration & hyperparameters
│   │   ├── state_encoder.py  # 16-dimensional spatial state vector encoder
│   │   └── types.py          # Pydantic data schemas & domain models
│   ├── data/                 # Dataset loaders & synthetic generation
│   │   ├── base_dataset.py   # Abstract dataset base class
│   │   ├── soccernet.py      # SoccerNet tracking dataset adapter
│   │   └── synthetic_generator.py # Synthetic tactical match trajectory generator
│   ├── evaluation/           # Scientific Evaluation & Metric Tracking
│   │   ├── evaluator.py      # ECE, Brier score, Log loss, F1, Confusion Matrix
│   │   └── experiment_tracker.py # JSON/CSV experiment logging engine
│   ├── explainability/       # Explainable AI (XAI)
│   │   ├── explainer.py      # Empirical model feature occlusion perturbation
│   │   └── goal_analyzer.py  # Goal sequence failure reconstruction
│   ├── models/               # PyTorch Deep Learning & Physics Models
│   │   ├── action_predictor.py # Action predictor wrapper & calibrated baseline
│   │   ├── pass_predictor.py   # Kinematic TTI pass trajectory model
│   │   └── temporal_model.py   # 2-layer PyTorch LSTM sequence model
│   ├── simulation/           # Counterfactual Tactical Simulation
│   │   ├── scenarios.py      # Preset tactical simulation scenarios
│   │   └── simulator.py      # Counterfactual What-If spatial delta engine
│   ├── tactical/             # Spatial Physics & Tactical Engine
│   │   ├── defensive_analysis.py # Direction-aware defensive collapse index
│   │   ├── goalkeeper.py     # Calibrated logistic sigmoid xG model
│   │   ├── homography.py     # OpenCV RANSAC homography matrix mapper
│   │   ├── pass_recommender.py # Time-To-Intercept pass ranking
│   │   ├── pitch.py          # Pitch coordinate transformation & mapping
│   │   └── video_analyzer.py # Automated tactical video sequence analyzer
│   ├── tracking/             # Persistent Data Export
│   │   └── exporter.py       # Parquet and JSONL tracking exporter
│   └── vision/               # Computer Vision & Video Processing
│       ├── ball_tracker.py   # Ball tracking & possession assignment
│       ├── detector.py       # YOLOv8/v11 player/ball object detection
│       ├── pose.py           # Player pose & body orientation estimation
│       ├── team_classifier.py# SigLIP + HSV K-Means jersey team identification
│       ├── tracker.py        # Hungarian MOT tracking & velocity estimation
│       └── video_processor.py# Full CV video annotation pipeline
├── configs/                  # Global YAML configuration files
│   └── config.yaml
├── docs/                     # Comprehensive Architectural & Technical Documentation
│   ├── API.md                # REST API reference specification
│   ├── ARCHITECTURE.md       # 4-tier system architecture blueprint
│   ├── EXPERIMENT_TRACKING.md# Experiment tracker & metric definitions
│   ├── LIMITATIONS.md        # Technical limitations & future roadmap
│   ├── PROJECT_ANALYSIS.md   # Architectural & statistical project analysis
│   └── RESEARCH_CONTRIBUTION.md # Research novelty & scientific contributions
├── experiments/              # Local experiment logs (JSON & CSV)
├── models/                   # Saved PyTorch model weights (`action_predictor.pt`)
├── tests/                    # Pytest test suite (37 passing unit & integration tests)
├── training/                 # Model training and evaluation scripts
├── uploads/                  # Video upload storage and generated annotated MP4 files
├── .env.example              # Environment variables template
├── pyproject.toml            # Build system and project metadata
└── requirements.txt          # Python dependencies
```

---

## 📜 Documentation Index

- 📘 [System Architecture Blueprint](docs/ARCHITECTURE.md) — 4-tier architecture, state encoding formulas, and model designs.
- 🌐 [REST API Reference](docs/API.md) — Endpoint specifications, query parameters, and example JSON payloads.
- 📈 [Experiment Tracking & Metrics](docs/EXPERIMENT_TRACKING.md) — ECE calibration, Brier score, and training run tracking.
- 🔬 [Research Contributions & Novelty](docs/RESEARCH_CONTRIBUTION.md) — Scientific innovations in sports intelligence and XAI.
- 📊 [Project Technical Analysis](docs/PROJECT_ANALYSIS.md) — Deep-dive statistical and architectural analysis.
- 🔮 [Limitations & Future Roadmap](docs/LIMITATIONS.md) — Current constraints and planned future advancements.
  

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14+-000000.svg)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4+-38B2AC.svg)](https://tailwindcss.com/)
[![Pytest 37 Passed](https://img.shields.io/badge/Pytest-37%20Passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)



---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
