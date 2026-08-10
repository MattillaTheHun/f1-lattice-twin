# Learning Notes

> Working log of concepts and decisions understood well enough to explain cold — not a code changelog (see `model_architecture.md` for that). Filled in as concepts are actually worked through, not dumped in wholesale.

---

## Architecture decisions — rationale I can defend out loud

**Why 2025 data is excluded from training.** 2026 F1 regulations changed the physical car (lighter chassis, active aero, new hybrid power split, overtake mode replacing DRS). A model trained on 2025 telemetry is learning a different vehicle. Analogy: this is the same problem as a hardware revision on a sensor platform — old calibration data doesn't transfer cleanly to new hardware. 2025 laps stay in the repo for EDA/visualization only.

**Why GRU over Transformer, with GBM as the honest baseline.** At ~360K rows for a single driver's season (before scaling to all drivers), a Transformer's attention mechanism doesn't have enough data to learn meaningful patterns — its inductive bias is weak at this scale, and it's O(n²) in sequence length for no payoff. A GRU's gating mechanism suits short-memory dynamics well (a braking event 0.5s ago matters, one 30s ago doesn't). Gradient-boosted trees on lagged features are a legitimate baseline, not a placeholder — if the GRU can't clearly beat GBM, the added complexity isn't earning its keep.

**Why the validation loop is chronological and "frozen."** Random-split validation on a time series leaks future context into training via shared rolling windows. The model is frozen before each new 2026 race weekend and scored only after — a genuine forward test, not a backtest. This is the single most defensible piece of the whole project when explaining it to someone technical.

**Why "driver" is a conditioning variable, not a forked model.** A GRU conditions on driver identity (like conditioning on an autonomous agent's control policy or config profile) rather than training a separate model per driver — same as how a fusion system doesn't run a separate inference stack per platform type. This is why training across multiple drivers is additive, not fragmenting.

**The Anduril translation, one paragraph.** Lattice fuses multi-domain sensor data into a real-time common operating picture, then recommends or autonomously tasks actions with a human authorizing the call. This project's state forecaster (predicted future entity state from fused telemetry) and pit classifier (predicted service/resupply need from operational context) are small, public-data versions of the same two problems: "what will this tracked thing do next" and "when does this asset need to come back."

---

## Concepts — understood well enough to teach back

### Kalman filtering / Gaussian data fusion
*Status: static case (measurement update) derived by hand. Dynamic/tracking case (predict step) not yet covered.*

**The static case — combining two independent readings of a fixed true value.**

Model each sensor as a Gaussian: reading = mean, sensor noise = standard deviation (σ). Two independent readings of the same true value combine via precision-weighted averaging, where precision = 1 / variance = 1 / σ² — **not** 1 / σ. This is the part that's easy to get backwards: variance is σ *squared*; the weighting ratio between two sensors is the ratio of their variances, not the ratio of their standard deviations.

Worked example: lab thermometer reads 21.0°C (σ = 0.1 → variance = 0.01), cheap thermometer reads 23.0°C (σ = 1.0 → variance = 1.0). Variance ratio = 1.0 / 0.01 = 100. The 2°C gap between readings splits by that ratio: 2 / 100 = 0.02. Combined estimate: 21.02°C — barely moved off the precise sensor, because "10x noisier" in σ terms means "100x less precise" in variance terms.

General formula (precision-weighted average):

```
combined_mean = (μ1/σ1² + μ2/σ2²) / (1/σ1² + 1/σ2²)
combined_variance = 1 / (1/σ1² + 1/σ2²)
```

Non-obvious result worth remembering: the combined estimate is *more confident* (smaller variance) than either individual sensor alone — even the noisy sensor's contribution can only help, never hurt, in this framework. This is the Kalman filter's "measurement update" step. The missing piece is the "predict" step: how the belief evolves *between* measurements via a dynamics model, before the next reading arrives and gets fused in the same way. Not yet covered.

### Bayesian optimization
*Status: not started — already have hands-on background from Kraft Heinz DOE work; revisit in the context of using the trained model as a simulator to search pit-strategy space.*

### PLS / PCA bridge to Kalman filtering
*Status: not started — the shared idea is combining information sources weighted by how much each should be trusted (covariance-weighted in PLS, variance-weighted in Kalman filtering).*

---

## Note on how this file is maintained

This doc is written and updated from the Cowork chat session (the one with the concept-teaching thread), not from Claude Code CLI — the terminal session has no visibility into these conversations, so asking it to backfill this file would produce something generic or wrong. Update it here, in the chat that actually has the context, then let it sit in the repo as the durable record.
