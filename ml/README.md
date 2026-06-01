# ML Pipeline: User-Product Interactions

This project turns the original notebook logic into a small Python pipeline for predicting next-period product demand.

The model uses a two-stage hurdle approach:

- Stage 1 predicts whether demand will be non-zero.
- Stage 2 predicts the positive purchase count.
- The final prediction combines classifier probability, regressor output, a simple baseline, and an optional blend.
- MLflow logs parameters, metrics, plots, models, and validation predictions.

## Structure

```text
ml/
|-- src/
|   |-- config.py
|   |-- data/bigquery_loader.py
|   |-- features/build_features.py
|   |-- split/snapshot_split.py
|   |-- metrics/metrics.py
|   |-- models/
|   |-- visualization/
|   `-- training/train_pipeline.py
|-- .env.example
|-- .gitignore
|-- requirements.txt
`-- README.md
```

Main entry point:

```text
ml/src/training/train_pipeline.py
```

## Setup

Create an environment and install dependencies:

```bash
cd ml
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy the example environment file:

```bash
copy .env.example .env
```

Then fill in values such as `GCP_PROJECT`, `MLFLOW_TRACKING_URI`, and `FEATURE_SET`.

## Run

From the `ml/` directory:

```bash
python -m src.training.train_pipeline
```

## Feature Sets

Use `FEATURE_SET` to compare runs with different feature groups:

- `base`: core numeric and categorical features.
- `funnel`: base features plus funnel/conversion features.
- `extended`: all engineered features.

Example:

```bash
set FEATURE_SET=base
python -m src.training.train_pipeline

set FEATURE_SET=extended
python -m src.training.train_pipeline
```

Each run logs `FEATURE_SET` to MLflow, so results can be compared directly.

