# FootballMind Technical Architecture Blueprint

## System Overview
**FootballMind** is a multimodal deep learning and spatial physics framework for predictive, tactical, and explainable football intelligence. The architecture follows a decoupled, 4-tier design:

```
+-----------------------------------------------------------------------------------+
|                            1. PERCEPTION TIER (Vision)                            |
|  Ultralytics YOLOv8/v11  -->  Hungarian MOT Tracker  -->  HSV Jersey Clustering   |
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
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                      4. PRESENTATION & API TIER (FastAPI/Next.js)                 |
|  FastAPI REST Server (8000)  -->  Next.js 14 Interactive Canvas Tactical Dashboard |
+-----------------------------------------------------------------------------------+
```

---

## Tier Specifications

### Tier 1: Perception Tier (`src/vision/`)
- **Detector (`FootballDetector`)**: Uses Ultralytics YOLOv8m/COCO to identify `player` and `sports ball` bounding boxes. Fallback mechanism provided for headless CPU demo execution (`HeuristicSyntheticDetectorFallback`).
- **Multi-Object Tracker (`FootballTracker`)**: Solves the linear sum assignment problem via the **Hungarian Algorithm** (`scipy.optimize.linear_sum_assignment`) over Euclidean distance matrices to maintain persistent track IDs and compute frame-to-frame velocity vectors $(v_x, v_y)$, speed, and heading direction.
- **Team Classifier (`TeamClassifier`)**: Crops the upper 40% torso region of detected player bounding boxes, applies an HSV green grass background mask ($35 \le H \le 85$) to isolate jersey fabric pixels, and clusters jersey colors into Home vs Away using Scikit-Learn KMeans ($k=2$).
- **Pitch Homography Mapper (`PitchMapper` / `HomographyMapper`)**: Computes a $3 \times 3$ perspective transformation matrix $\mathbf{H}$ using OpenCV RANSAC mapping pixel points to top-down pitch dimensions ($105\text{m} \times 68\text{m}$).

---

### Tier 2: State Encoding Tier (`src/core/`)
- **State Encoder (`FootballStateEncoder`)**: Converts raw `FrameTacticalState` objects into a 16-dimensional standardized numerical feature vector $\mathbf{x} \in \mathbb{R}^{16}$ per player:

| Index | Feature Name | Formula / Range |
|-------|--------------|-----------------|
| 0 | Player X Normalized | $x / 105.0$ |
| 1 | Player Y Normalized | $y / 68.0$ |
| 2 | Velocity X | $v_x$ (m/s) |
| 3 | Velocity Y | $v_y$ (m/s) |
| 4 | Speed Normalized | $\text{speed} / 10.0$ |
| 5 | Heading Direction | $\theta_{\text{dir}} / 360.0$ |
| 6 | Ball X Normalized | $x_{\text{ball}} / 105.0$ |
| 7 | Ball Y Normalized | $y_{\text{ball}} / 68.0$ |
| 8 | Nearest Teammate Distance | $d_{\text{tm}} / 50.0$ |
| 9 | Nearest Opponent Distance | $d_{\text{opp}} / 50.0$ |
| 10 | Defensive Pressure Level | $\max(0, 1 - d_{\text{opp}} / 10.0)$ |
| 11 | Teammate Density (8m) | $N_{\text{tm, 8m}} / 10.0$ |
| 12 | Opponent Density (8m) | $N_{\text{opp, 8m}} / 10.0$ |
| 13 | Available Pitch Space | $\min(d_{\text{opp}}, 15.0) / 20.0$ |
| 14 | Distance to Goal | $d_{\text{goal}} / 105.0$ |
| 15 | Ball Proximity | $d_{\text{ball}} / 50.0$ |

---

### Tier 3: Intelligence & Physics Engine (`src/models/`, `src/tactical/`, `src/simulation/`, `src/explainability/`)

#### 1. Temporal Action Prediction (`ActionPredictor` & `FootballTemporalLSTM`)
- **PyTorch Architecture**: 2-layer LSTM with hidden dimension size 64, taking sliding window sequences of shape $[B, 10, 16]$ and projecting to 8 action classes (PASS, DRIBBLE, SHOT, CROSS, TACKLE, CARRY, HOLD, CLEARANCE).
- **Calibrated Multinomial Softmax Baseline**: Temperature-scaled ($\tau = 1.2$) multinomial softmax over spatial physics logits:
  $$\mathbf{P}(Y = a_i \mid \mathbf{x}) = \frac{e^{z_i / \tau}}{\sum_{j=1}^8 e^{z_j / \tau}}$$

#### 2. Time-To-Intercept (TTI) Kinematic Pass Prediction (`PassPredictor`)
- **Ball Velocity Drag Equation**:
  $$v_{\text{ball}}(t) = v_0 e^{-\mu t} \implies t_{\text{ball}}(d) = -\frac{1}{\mu} \ln\left(1 - \frac{\mu d}{v_0}\right)$$
- **Opponent Time-to-Intercept**:
  $$t_{\text{opp}} = t_{\text{reaction}} + \frac{d_{\text{proj}}}{v_{\text{sprint}}}$$
- **Success Probability**:
  $$P(\text{success}) = \frac{1}{1 + e^{-3.0 \cdot (t_{\text{opp}} - t_{\text{ball}} - 0.15)}}$$

#### 3. Logistic Sigmoid Expected Goals Model (`GoalkeeperAnalyzer`)
- **Logistic xG Equation**:
  $$\text{xG} = \frac{1}{1 + e^{-(\beta_0 + \beta_1 d_{\text{goal}} + \beta_2 \theta_{\text{shot}} + \beta_3 n_{\text{def}}) reviews}}$$
  where $\beta_0 = -0.80$, $\beta_1 = -0.11$, $\beta_2 = +1.80$, $\beta_3 = -0.55$.

#### 4. Direction-Aware Defensive Collapse Index (`DefensiveAnalyzer`)
- Calculates CB-LB and CB-RB lateral gaps, midfield line distance, Zone 14 unblocked passing lane exposure, unmarked attackers in attacking third, and penalty box occupation.

#### 5. Empirical Model Feature Occlusion Perturbation (`ExplainabilityEngine`)
- Measures exact probability drops when masking each input feature vector dimension $i$:
  $$\Delta P_i = P_{\text{baseline}}(y_{\text{target}}) - P_{\text{masked\_feature\_i}}(y_{\text{target}})$$

---

### Tier 4: Presentation & API Tier (`backend/`, `frontend/`)
- **FastAPI Backend (Port 8000)**: Asynchronous REST endpoints exposing tracking, predictions, pass rankings, defensive danger indices, What-If simulation counterfactuals, explainability attributions, and scientific evaluation metrics.
- **Next.js 14 Dashboard (Port 3000)**: Responsive UI featuring interactive HTML5 Canvas tactical pitch rendering, drag-and-drop What-If player simulation, real-time probability distribution cards, and model calibration badges.
