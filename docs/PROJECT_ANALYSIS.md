# FootballMind: Project Analysis & Architecture Plan

## 1. Executive Summary & Core Research Objective
**Project Title**: FootballMind: A Multimodal Deep Learning Framework for Predictive, Tactical and Explainable Football Intelligence  
**Core Research Question**: Can an AI system move beyond recognizing football events to predicting player decisions, recommending optimal actions, simulating tactical alternatives, and explaining why those decisions lead to success or failure?

FootballMind is designed as an end-to-end, production-grade football intelligence platform. It ingests broadcast/tactical football video or tracking streams and outputs real-time visual perception, pitch-space coordinate mappings, temporal tactical state representations, predictive action/pass models, what-if counterfactual scenario simulations, explainable AI attribution cards, and an interactive modern web dashboard.

---

## 2. Current Repository Inspection
- **Directory**: `C:\Users\Admin\Downloads\ball`
- **Current State**: Initialized workspace.
  - `.agents/skills/`: Higgsfield brandkit/generate/websites/etc. skills installed.
  - `skills-lock.json`: Active skills locking file.
- **Environment**:
  - Python: `3.10.9` (PyTorch `2.13.0+cpu`, OpenCV `4.11.0`, Ultralytics `8.x`, FastAPI `0.111.0`, Pydantic v2).
  - Node.js: `v22.17.1`, npm: `10.9.2`.
- **Existing Functionality**: Clean workspace ready for modular scaffolding.
- **Missing Functionality**: Complete Python backend architecture (`src/`, `backend/`, `training/`, `tests/`), Next.js dashboard (`frontend/`), documentation (`docs/`), CLI wrapper, and test suites.

---

## 3. Proposed System Architecture

```
                                +-----------------------------+
                                |     FOOTBALL VIDEO / STREAM |
                                +--------------+--------------+
                                               |
                                               v
                                +-----------------------------+
                                |       VIDEO INGESTION       |
                                |     (src/vision/video.py)   |
                                +--------------+--------------+
                                               |
                                               v
                                +-----------------------------+
                                |  COMPUTER VISION PERCEPTION |
                                |  - Detection (YOLOv8/11)    |
                                |  - Tracking (ByteTrack/SORT)|
                                |  - Team & Color Clustering  |
                                |  - Pose / Orientation       |
                                +--------------+--------------+
                                               |
                                               v
                                +-----------------------------+
                                |  HOMOGRAPHY & PITCH MAPPING |
                                |   (Camera -> Pitch [105x68])|
                                +--------------+--------------+
                                               |
                                               v
                                +-----------------------------+
                                |   FOOTBALL STATE ENCODER    |
                                | - Velocity/Acc/Spatial Rel. |
                                | - Dynamic Tactical Graph    |
                                +--------------+--------------+
                                               |
        +--------------------------------------+--------------------------------------+
        |                                      |                                      |
        v                                      v                                      v
+---------------+                      +---------------+                      +---------------+
|  PREDICTION   |                      |   TACTICAL    |                      |    WHAT-IF    |
|    ENGINE     |                      |    ENGINE     |                      |   SIMULATOR   |
| - Next Action |                      | - Def. Collapse|                      | - Position Mod|
| - Receiver    |                      | - Space Score |                      | - Recalc xG   |
| - Best Pass   |                      | - Pitch Control|                      | - Delta Index |
+-------+-------+                      +-------+-------+                      +-------+-------+
        |                                      |                                      |
        +--------------------------------------+--------------------------------------+
                                               |
                                               v
                                +-----------------------------+
                                |    EXPLAINABILITY ENGINE    |
                                | - SHAP / Feature Attrib.    |
                                | - Goal Cause Reconstruction |
                                +--------------+--------------+
                                               |
                                               v
                                +-----------------------------+
                                |       FASTAPI BACKEND       |
                                | (API Routes & WS Stream)    |
                                +--------------+--------------+
                                               |
                                               v
                                +-----------------------------+
                                |    NEXT.JS 14 DASHBOARD     |
                                | (Tactical Pitch & Analytics)|
                                +-----------------------------+
```

---

## 4. Implementation Roadmap (Phases 0 - 15)

- **PHASE 0**: Repository analysis, environment verification, and architectural blueprints (`docs/PROJECT_ANALYSIS.md`).
- **PHASE 1**: Project scaffolding (`pyproject.toml`, `requirements.txt`, directory hierarchy, package initialization).
- **PHASE 2**: Data ingestion & synthetic benchmark dataset generator (`src/data/`).
- **PHASE 3**: Computer vision perception pipeline (detector, tracker, team classifier, pose estimation).
- **PHASE 4**: Persistent Multi-Object Tracking & Parquet/JSONL state export.
- **PHASE 5**: Homography, pitch detection, and normalized coordinate transform (`src/tactical/pitch.py`, `homography.py`).
- **PHASE 6**: Unified Football State Encoder (`src/core/state_encoder.py`) & vector representations.
- **PHASE 7**: Temporal Intelligence & Action/Pass/Receiver Prediction models (XGBoost baseline + PyTorch LSTM/GRU).
- **PHASE 8**: Tactical Intelligence & Defensive Collapse Index / xG / Best Pass Recommender.
- **PHASE 9**: Counterfactual What-If Tactical Simulator (`src/simulation/`).
- **PHASE 10**: Explainability Engine & Goal Sequence Reconstructor (`src/explainability/`).
- **PHASE 11**: FastAPI Backend APIs (`backend/`).
- **PHASE 12**: Next.js 14 Dashboard frontend with dark modern UI design (`frontend/`).
- **PHASE 13**: Unit & Integration test suite (`tests/` with `pytest`).
- **PHASE 14**: Pipeline optimization (batching, CPU fallback, frame skipping, caching).
- **PHASE 15**: Full academic/engineering documentation (`docs/` & `README.md`).
