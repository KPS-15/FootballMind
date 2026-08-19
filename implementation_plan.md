# Implementation Plan - FootballMind: Multimodal Deep Learning Framework for Football Intelligence

"FootballMind: A Multimodal Deep Learning Framework for Predictive, Tactical and Explainable Football Intelligence" is a complete production-grade football analytics platform. It ingests video/tracking streams, processes vision detections, projects player coordinates onto a 105x68 pitch, encodes temporal tactical states, predicts player intentions & best pass opportunities, simulates counterfactual tactical scenarios, provides explainable AI attribution, and exposes these insights via a FastAPI backend and Next.js 14 interactive dashboard.

## User Review Required

> [!IMPORTANT]
> The system requires both Python 3.10 and Node.js environments. Detections run locally via Ultralytics YOLOv8/v11 with OpenCV & ByteTrack fallback. When model weights or live video feeds are absent, FootballMind seamlessly operates in **Demo/Baseline Mode** with synthetic tracking generators and baseline models, adhering strictly to Rule 40 (No fake hardcoded results; real feature pipelines with baseline math & ML models).

> [!NOTE]
> Frontend aesthetics will adopt a modern dark football analytics UI design system with responsive visual tactical pitch, dynamic pass lane vectors, probability charts, and what-if sliders.

---

## Proposed System & Phase Roadmap

### Phase 1: Project Scaffolding & Configuration
- Create core directory layout (`src/`, `backend/`, `frontend/`, `training/`, `models/`, `datasets/`, `tests/`, `configs/`, `docs/`, `scripts/`).
- Define `pyproject.toml`, `requirements.txt`, `.env.example`, `.gitignore`, and `configs/config.yaml`.
- Set up logging, device auto-detection (`CUDA` vs `CPU`), and CLI entry point (`footballmind`).

### Phase 2: Data Ingestion & Synthetic Benchmark Generator
- Define data structures for tracking frames, player objects, ball states, and tactical metrics.
- Implement dataset adapters (`src/data/base_dataset.py`, `soccernet.py`, `synthetic_generator.py`).
- Generate realistic match tracking trajectories (22 players + ball, 500+ frames) for testing and demo mode.

### Phase 3 & 4: Computer Vision & Multi-Object Tracking
- Detector module (`src/vision/detector.py`) using Ultralytics YOLO with bounding box center math and confidence scoring.
- Multi-Object Tracker (`src/vision/tracker.py`) managing persistent `track_id`, position smoothing, velocity, acceleration, speed, and heading direction.
- Color/Team classifier (`src/vision/team_classifier.py`) clustering HSV jersey colors into Home/Away teams.
- Ball Tracker (`src/vision/ball_tracker.py`) handling trajectory interpolation & possession association.
- Parquet/JSONL export pipeline (`src/tracking/exporter.py`).

### Phase 5: Pitch Mapping & Homography
- Coordinate Transformer (`src/tactical/pitch.py`, `homography.py`).
- Map pixel coordinates `(x_pixel, y_pixel)` to normalized pitch coordinates `[0..105, 0..68]`.
- Implement camera-to-pitch homography matrix solver with calibration point fallbacks.

### Phase 6: Unified Football State Encoder
- Numerical feature encoder (`src/core/state_encoder.py`).
- Calculate spatial relationships for each time step: nearest defender distance, teammate support density, passing lane vectors, available pitch space (Voronoi/Gaussian density), and pressure index.

### Phase 7: Temporal ML Predictions
- Action Predictor (`src/models/action_predictor.py`): Predicts PASS, DRIBBLE, SHOT, CROSS, TACKLE, CARRY, HOLD, CLEARANCE.
- Pass & Receiver Predictor (`src/models/pass_predictor.py`): Evaluates target receiver ID, target landing point, and completion probability.
- Temporal Models (`src/models/temporal_model.py`): Features XGBoost/RandomForest baselines alongside PyTorch LSTM/GRU network architecture.

### Phase 8: Tactical Intelligence, Best Pass & Defensive Danger
- Best Pass Recommender (`src/tactical/pass_recommender.py`): Scores candidate passes = `Success Prob * Attacking Advantage * Space Created`.
- Defensive Collapse Index (`src/tactical/defensive_analysis.py`): Computes CB-LB gap, CB-RB gap, midfield exposure, unmarked attackers, and overall danger index (0-100%).
- Goal Probability (xG) & Goalkeeper positioning estimator (`src/tactical/goalkeeper.py`).

### Phase 9: Counterfactual What-If Tactical Simulator
- Interactive simulator engine (`src/simulation/simulator.py`, `scenarios.py`).
- Accepts modified player coordinates (e.g. "Move Left-Back 3.2m inward"), re-encodes state features, recalculates passing lanes, defensive collapse index, and xG, outputting delta impact reports.

### Phase 10: Explainability & Goal Sequence Reconstructor
- Explainable AI Engine (`src/explainability/explainer.py`): SHAP-inspired feature attributions, key factor ranking, and decision breakdown cards.
- Goal Sequence Analyzer (`src/explainability/goal_analyzer.py`): Reconstructs 5-10s pre-goal sequence to identify primary structural breakdown, critical timestamps, and recommended counter-actions.

### Phase 11: FastAPI Backend API
- FastAPI Application (`backend/main.py`, `routes/`, `schemas/`, `services/`).
- Endpoints:
  - `POST /api/video/upload` & `POST /api/video/process`
  - `GET /api/match/{id}`, `GET /api/tracking/{id}`
  - `GET /api/prediction/{id}`, `GET /api/tactical/{id}`
  - `GET /api/recommendations/{id}`, `POST /api/simulation`
  - `GET /api/explanation/{id}`, `GET /api/evaluation/{id}`
  - WebSocket `/ws/match/{id}` for live simulation/video updates.

### Phase 12: Next.js 14 Interactive Dashboard
- Modern dark-themed dashboard frontend in `frontend/`.
- Interactive SVG/Canvas top-down 2D Tactical Pitch with player nodes, velocity vectors, Voronoi control, passing lane heatmaps, and what-if drag controls.
- Real-time prediction cards, Best Pass breakdown, Defensive Collapse Index metrics, What-If scenario sliders, and SHAP explainability cards.

### Phase 13 - 15: Testing, Optimization & Comprehensive Documentation
- Pytest suite (`tests/`) covering vision parsing, tracking math, homography, state encoding, ML models, simulator deltas, and FastAPI endpoints.
- Optimizations: frame skipping, batch inference, model caching.
- Documentation suite (`docs/ARCHITECTURE.md`, `DATASET.md`, `MODELS.md`, `TRAINING.md`, `API.md`, `EXPERIMENTS.md`, `RESEARCH_CONTRIBUTION.md`, `LIMITATIONS.md`, `DEPLOYMENT.md`, `README.md`).

---

## Verification Plan

### Automated Testing
- `pytest tests/ -v` (Unit + integration tests for state encoder, pitch transform, models, simulator, backend endpoints).
- `py -m footballmind demo` (CLI synthetic demonstration test).

### Manual Verification & Dashboard Launch
- Start FastAPI backend (`uvicorn backend.main:app --port 8000`).
- Start Next.js dashboard (`cd frontend && npm run dev`).
- Load match session, inspect real-time pitch view, test what-if position modifications, verify explainability cards and API endpoint responses.
