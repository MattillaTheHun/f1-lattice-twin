# Model Architecture — F1 Lattice Twin

> **Status:** Design doc — no implementation yet.
> Last updated: 2026-08-09

---

## 0. Framing

The system answers one question in real time:

> "Given everything the car's sensors have reported up to now, what will the car's state look like N timesteps from now — and how likely is it to pit next lap?"

Two coupled prediction targets:

1. **State forecaster** — multivariate regression: predict `Speed`, `X`, `Y` at horizon T+N (N ≈ 10–40 steps, ~0.5–2 s at 20 Hz sampling)
2. **Service-window classifier** — binary/probabilistic: P(pit in next K laps), K ∈ {1, 2, 3}

`Driver` (e.g. NOR, PIA) is a **conditioning variable**, not a trained weight. It identifies which agent's control policy generated the telemetry stream — analogous to an autonomous platform's config profile or control policy ID in a multi-agent system. Different drivers produce systematically different throttle/brake signatures at the same track location; the model conditions on driver identity to avoid conflating those distributions without learning a separate model per driver. In a Lattice-style system, this corresponds to tagging each entity track with its platform type or operator profile so the prediction layer can adjust expected behaviour without retraining for each new asset.

---

## 0.5. Anduril Translation

Anduril's Lattice platform is described publicly as a multi-domain sensor fusion system that ingests heterogeneous data streams into a unified, real-time common operating picture, then enables autonomous tasking and human-authorized action at machine speed. This project is a tractable, data-rich instantiation of that same architecture. The fused telemetry pipeline — merging 20 Hz car sensors with asynchronous weather feeds via timestamp-aligned merge — mirrors the sensor-fusion layer that Lattice runs across ISR feeds, radar, ADS-B, and ground sensors. The state forecaster (predicting `Speed`, `X`, `Y` at horizon T+N) maps directly onto Lattice's predicted future entity state: given what all sensors report about an asset now, where will it be and how will it be behaving N seconds from now? The pit-stop classifier maps onto predicted service/resupply need: a probabilistic signal, issued in advance, that an asset must return to a controlled zone — analogous to a drone returning to rearm, a UGV returning to recharge, or a logistics vehicle requiring maintenance before its next sortie. The human-authorized action layer is implicit here in the strategy engineer who acts on the pit window probability, exactly as an operator acts on a Lattice-surfaced tasking recommendation without removing human judgment from the loop.

---

## 1. Regulation Boundary and Data Partitioning

### Why the boundary matters

The 2026 regulations introduced wholesale changes: lighter chassis (~30 kg reduction), fully redesigned active aerodynamics (front/rear moveable elements replacing passive DRS), and a new hybrid PU architecture with a 50/50 ICE-ERS power split and a discrete "overtake mode" replacing the old DRS zone activations. These changes alter every meaningful physical relationship in the telemetry — drag curves, throttle response, cornering energy distribution, braking points. A model trained on 2025 data is learning a different car.

**Rule: 2025 data is for EDA and pipeline validation only. All training uses 2026-season races exclusively.**

### Train / validation / test split

```
2026 Rounds 1–9   (Australia → Great Britain excl. last 2)  →  Training corpus
2026 Rounds 10–11 (last 2 before Netherlands)               →  In-season hold-out validation
──────────────────────────────────── model frozen here ─────────────────────────────────────────
2026 Round 12     Netherlands (Aug 21–23)                   →  Unseen future test #1
2026 Round 13     Italy/Monza (Sep 4–6)                     →  Unseen future test #2
2026 Round 14     Spain/Madrid (Sep 11–13)                  →  Unseen future test #3
     ...and each subsequent round as it completes
```

**Within a session, splits are always chronological** — never random. Random shuffling of a time series allows future data to leak into training samples via shared rolling-window context.

- For within-race splits: first 70% of laps → train, last 30% → validation.
- For cross-race generalisation: leave-one-race-out cross-validation across rounds 1–9, using round 10–11 as the final pre-deployment validation gate.

### Expected 2026 data volume

Each race session: ~20 drivers × ~55 laps × ~600 telemetry rows/lap ≈ **660,000 rows/race**.

For a single driver across 11 races: ~11 × 55 × 600 ≈ **363,000 rows**.
For all drivers across 11 races: **~7.3 M rows**.

This matters enormously for model selection.

---

## 2. Model Selection Rationale

### The scale argument

At 363K–7.3M rows with 8–15 features and sequences of ~600 steps, the architecture space narrows quickly when reasoned from first principles rather than trend-chasing.

| Property | Kalman Filter (CV) | Windowed GBM | Small GRU/LSTM | Transformer |
|---|---|---|---|---|
| Sample efficiency | N/A — no training | High | Medium | Low — needs millions of tokens to learn attention patterns |
| Training time (CPU) | None | Minutes | Hours | Days without GPU |
| Sequence memory | One-step predict-update | Explicit lag features | Implicit hidden state | Full context via attention |
| Interpretability | Full — closed-form equations | SHAP, feature importance | Partial (gate activations) | Low |
| Overfitting risk at this scale | None | Low–medium | Medium | High |
| Hyperparameter sensitivity | Low (process/obs noise Q, R) | Low | Medium | High |
| Cold-start (1 race of data) | Immediate | Works well | Marginal | Fails |
| Handles throttle/brake discontinuities | Poorly — assumes smooth dynamics | Well | Well | Well |

**Kalman filter (constant-velocity model):** The classical technique underlying most production sensor-fusion systems — GPS/INS fusion, radar track management, missile guidance, autonomous vehicle localisation. A CV-Kalman filter propagates a state vector `[X, Y, vX, vY]` forward with a linear motion model and corrects it when a new observation arrives. It requires no training data and can be initialised from a single lap. Its performance establishes the **non-learned floor**: any ML model that cannot clearly beat a CV-Kalman on position error has not justified its existence. Expected failure modes here are cornering (CV assumes straight-line motion; a car at Eau Rouge is not moving in a straight line) and braking zones (instantaneous deceleration violates the smooth-dynamics assumption). These failure modes are exactly where GRU and GBM should pull ahead — and if they don't, that is the most important finding of Phase 1.

**Transformer verdict: ruled out.** Self-attention scales O(n²) with sequence length. More importantly, transformers need substantial data to learn meaningful positional encodings and attend across context — their inductive bias is weak for small structured sequences. At 600-step laps with 8–15 channels, you are not in the regime where a transformer earns its cost.

**Recommended primary model: GRU** with a single hidden layer (64–128 units). GRUs handle variable-length sequences naturally, capture temporal dependencies without explicit lag engineering, train stably at this data volume, and are small enough to retrain after each race weekend (~minutes on CPU, seconds on GPU). The gating mechanism is better matched to the short-memory nature of car-state dynamics (a braking event ~0.5 s ago is relevant; one 30 s ago is not) than a full transformer with global attention.

**Recommended baseline model: XGBoost or LightGBM with lagged features.** This is not a consolation — windowed gradient boosting is highly competitive with RNNs on tabular time-series at this scale. It establishes an honest upper bound for "how much does the sequential architecture actually add?" If the GRU cannot clearly beat an XGBoost baseline, the GRU is not justified. The baseline must be built first.

**Decision rule:** Deploy whichever model wins on held-out validation RMSE for position error and speed error. Default to GBM if the difference is marginal (< 5% RMSE reduction) — simpler is more trustworthy in a live scoring context.

---

## 3. Feature Engineering

### Input window

A rolling window of **W timesteps** (default W = 20, ~1 s of history at 20 Hz) is slid across the session with stride 1.

### Raw telemetry channels (per timestep in window)

| Feature | Unit | Notes |
|---|---|---|
| `Speed` | km/h | Direct sensor |
| `RPM` | rev/min | Direct sensor |
| `Throttle` | % (0–100) | Direct sensor |
| `Brake` | 0/1 | Binarised |
| `X`, `Y` | m (track coords) | Position data |
| `TrackTemp` | °C | Weather-fused |
| `AirTemp` | °C | Weather-fused |
| `Distance` | m into lap | Cumulative |

### Derived features (computed from window)

| Feature | Derivation |
|---|---|
| `dX`, `dY` | First-difference of X, Y — velocity components |
| `Speed_delta` | First-difference of Speed — longitudinal acceleration proxy |
| `Curvature` | \|dX × dY\| / \|v\|³ — radius-of-curvature estimate from XY trajectory |
| `Lateral_G_proxy` | Speed² × Curvature — centripetal acceleration estimate |
| `Throttle_rolling_mean` | Mean throttle over window — fuel/energy demand signal |
| `Brake_leading_edge` | Binary: Brake[t] > Brake[t-1] — braking initiation flag |
| `RPM_delta` | First-difference of RPM — gear change / overtake mode proxy |

### Metadata features (constant per lap, broadcast across window)

| Feature | Notes |
|---|---|
| `TyreAge` | Laps on current compound (for pit classifier) |
| `StintLength` | Laps in current stint |
| `LapNumber` | Absolute lap in race |
| `TyreCompound` | Soft / Medium / Hard — one-hot encoded |
| `GapAhead_s` | Gap to car ahead (from timing data) |
| `GapBehind_s` | Gap to car behind |

### Prediction targets

For state forecasting:
- `Speed[T+N]`, `X[T+N]`, `Y[T+N]`

For pit classifier:
- `PitNextLap` (binary), `PitIn2Laps`, `PitIn3Laps`

---

## 4. Architecture Specification

### 4a. State Forecaster (GRU path)

```
Input: [Batch, W=20, Features=15]
         │
    ┌────▼────────────────────┐
    │  GRU Layer 1            │
    │  hidden_size = 128      │
    │  dropout = 0.2          │
    └────┬────────────────────┘
         │  hidden state h_T
    ┌────▼────────────────────┐
    │  Linear(128 → 64)       │
    │  ReLU                   │
    └────┬────────────────────┘
         │
    ┌────▼────────────────────┐
    │  Linear(64 → 3 × N)     │   ← 3 targets × N horizon steps
    └────┬────────────────────┘
         │
    Output: [Batch, N, 3]  (Speed, X, Y)
```

Multi-step outputs are predicted jointly rather than autoregressively at training time to avoid error accumulation during the training signal — but at inference they can be unrolled autoregressively if real-time sensor updates arrive step-by-step.

### 4b. State Forecaster (GBM path — baseline)

Flatten the input window into a 1D feature vector of shape `[W × n_features]` (explicit lag features at offsets t, t-1, ..., t-W+1). Train separate XGBoost regressors for each target (Speed, X, Y) at each horizon step N. Three regressors × N horizon steps = 3N total models, but they are trivially parallelised and each trains in seconds.

### 4c. Pit-Stop Service-Window Classifier

```
Input: metadata features (TyreAge, StintLength, LapNumber,
       GapAhead, GapBehind, TyreCompound, last-N-laps LapTime delta)
         │
    LightGBM binary classifier (one per K ∈ {1, 2, 3})
         │
    Output: P(pit within K laps)  ∈ [0, 1]
```

Framed identically to autonomous-asset resupply prediction: given current resource state (tyre degradation proxy = tyre age × compound wear rate) and operational context (gaps, lap time trajectory), predict when the asset needs to return to base. The Kaggle Playground S6E5 pit-prediction competition uses these exact features and serves as both a feature reference and a sanity-check benchmark for classifier performance.

The classifier shares no weights with the state forecaster — it is a parallel head, not a joint model. This keeps failure modes isolated.

---

## 5. Training Protocol

### Loss functions

- State forecaster: **Huber loss** (δ=1.0) per target, summed. Huber is preferred over MSE because telemetry contains outliers (sensor glitches, SC/VSC laps) that MSE would over-weight.
- Pit classifier: **Binary cross-entropy** with positive-class weighting to correct for class imbalance (pit laps are ~5–8% of all laps).

### Optimiser

- GRU: AdamW, lr=1e-3, weight decay=1e-4, cosine annealing LR schedule.
- GBM: default LightGBM with early stopping on validation RMSE (patience=50 rounds).

### Normalisation

Per-feature standardisation (z-score) fit on training data only. Scaler is serialised alongside the model weights so inference uses the same statistics. Position coordinates (X, Y) are normalised per circuit — a Monza coordinate is meaningless at Monaco.

### Regulation-safe data loading

A `regulation_boundary` flag in `build_dataset.py` will gate year at `>= 2026`. Any 2025 rows that accidentally enter the pipeline raise an assertion error at training time, not a silent inclusion.

---

## 6. Evaluation Metrics

### State forecaster

| Metric | Formula | Notes |
|---|---|---|
| Speed RMSE | √mean((ŷ−y)²) in km/h | Primary speed metric |
| Position Error | √((X̂−X)²+(Ŷ−Y)²) in metres | Euclidean 2D error |
| MAE (all targets) | mean(\|ŷ−y\|) | Interpretable alongside RMSE |
| Horizon degradation curve | Error vs N plot | Key diagnostic — shows where model diverges |

The horizon degradation curve is the primary model health report: it should show error growing sub-linearly with N. A sharp cliff at any horizon step signals a failure mode (e.g. model loses track of a braking zone).

### Pit classifier

| Metric | Notes |
|---|---|
| AUC-ROC | Threshold-free, handles imbalance |
| Precision@K (K=1,2,3 laps) | Operational cost of false alarms |
| Recall@K | Cost of missed pit windows |
| Brier Score | Calibration of probability outputs |

---

## 7. Phased Implementation Plan

### Phase 1 — Proof of Concept (single driver, single 2026 race)

**Goal:** Close the loop end-to-end. Prove the pipeline can train, predict, and score.

- Scope: NOR fastest lap per session, Round 1 (2026 Australian GP)
- Model: XGBoost baseline only
- Targets: `Speed[T+1]` only (next-timestep, not multi-step)
- Output: RMSE on held-out 30% of the lap, horizon degradation plot
- Deliverable: `src/models/train_baseline.py`, `notebooks/02_baseline_forecast.ipynb`

**Gate to Phase 2:** Baseline RMSE on Speed < 5 km/h at T+1, position error < 10 m.

### Phase 2 — Multi-Race, Multi-Driver 2026 Dataset

**Goal:** Train across rounds 1–9, validate on rounds 10–11, compare GRU vs GBM.

- Scope: All available drivers, all 2026 races through Round 9
- Models: XGBoost baseline + GRU challenger
- Targets: Speed, X, Y at horizons N ∈ {1, 5, 10, 20, 40} steps (0.05–2 s)
- Add: pit-stop classifier head (LightGBM)
- Add: circuit-specific coordinate normalisation
- Deliverable: `src/models/train_gru.py`, `src/models/train_pit_classifier.py`, trained model artifacts in `models/`

**Gate to Phase 3:** GRU beats GBM by ≥ 5% on position error at N=20; pit classifier AUC ≥ 0.80.

### Phase 3 — Live Validation Loop

**Goal:** Score the frozen Phase 2 model against each new 2026 race as it completes.

Protocol:
1. **Before each race weekend:** freeze model weights, log commit hash + freeze timestamp.
2. **After the race:** run `src/build_dataset.py` to pull new session data.
3. **Score:** run `src/evaluate_frozen.py` — compute metrics against ground truth, append to `data/live_validation_log.csv`.
4. **Do not retrain** until a predefined schedule (e.g. every 4 rounds). Retraining mid-season on a whim invalidates the validation experiment.

This mirrors a production ML deployment: the model makes predictions before outcomes are known, and is scored afterwards. It is the only honest test of generalisation to genuinely unseen conditions (new track layouts, evolving car setups, weather variation across rounds).

---

## 8. Stretch Goal — Autonomous Asset Analogy

The pit-stop classifier maps cleanly onto a broader class of problems:

> "Given current resource state and operational context, predict when an autonomous asset (vehicle, drone, robot) needs to return for resupply or maintenance."

| F1 Domain | Autonomous Asset Domain |
|---|---|
| Tyre age | Battery state / consumable inventory |
| Stint length | Time since last service |
| Lap time delta vs baseline | Performance degradation vs nominal |
| Gap to competitor | Mission urgency / deadline pressure |
| Pit window | Service window / resupply opportunity |

The classifier architecture is identical; only the feature labels change. This framing makes the system transferable and is worth documenting as a separate design note once Phase 2 is complete.

---

## 9. Open Questions (to resolve in Phase 1)

1. **Safety car laps** — telemetry during SC/VSC periods breaks the normal speed-distance relationship. Should these laps be masked out of training, or will the model learn to handle them as a distinct regime?
2. **Sector-boundary effects** — FastF1 telemetry occasionally has small gaps at sector boundaries. How robust is the rolling-window approach to missing rows?
3. **Circuit generalisation** — does a model trained on Monza/Spa generalise to Monaco's 3.3 km layout? Or is per-circuit coordinate normalisation insufficient and a circuit embedding needed?
4. **Overtake mode telemetry** — does the 2026 PU's overtake mode appear as a discrete RPM/throttle spike? If so, it may need explicit flagging as a feature to avoid confusing the forecaster.
