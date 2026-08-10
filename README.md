# F1 Lattice Twin

An F1 telemetry sensor-fusion digital twin built on [FastF1](https://docs.fastf1.dev/) data.

## Overview

This project fuses multi-channel telemetry streams (speed, throttle, brake, DRS, tyre temps, sector splits) into a predictive digital twin for Formula 1 race strategy. Core targets include pit-stop timing optimization and lap-time modeling across variable track/weather conditions.

## Structure

```
data/        Raw and processed telemetry data
src/         Source modules (ingestion, feature engineering, models)
notebooks/   Exploratory analysis and prototyping
models/      Saved model artifacts
.github/     CI workflows
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
