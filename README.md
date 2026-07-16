# IEEE-CIS Fraud Detection Workspace

This branch turns the repository into a small, repeatable fraud-detection workflow for the IEEE-CIS Kaggle dataset. It covers loading the raw data, exploring fraud patterns, selecting useful features, and training several baseline fraud classifiers.

## What This Branch Adds

The work is split into four main areas.

## 1. Data Loading And Preprocessing

[fraud_preprocessing.py](fraud_preprocessing.py) contains the shared data-loading and numeric preprocessing code.

It can:

- Load the IEEE-CIS train and test CSV files from `data/`.
- Join transaction and identity files on `TransactionID`.
- Normalize identity column names such as `id-01` to `id_01`.
- Mark whether a transaction has matching identity data.
- Cast known categorical columns to categorical types.
- Encode mixed raw columns into a numeric model matrix.
- Convert boolean-like values such as `T` and `F` into numeric flags.
- One-hot encode lower-cardinality categories.
- Frequency encode higher-cardinality categories.
- Add missing-value indicator columns.
- Impute numeric missing values using fitted medians.
- Save and reload the fitted preprocessor so train, validation, and test data use the same encoded columns.

Example:

```bash
conda run -n fraud python fraud_preprocessing.py
```

By default this uses the small example subset in `data/example_subset/` and writes outputs to `outputs/example_preprocessing/`.

## 2. Dataset Exploration

[fraud_exploration.py](fraud_exploration.py) adds exploratory analysis for fraud cases and the full dataset.

It can:

- Summarize confirmed fraud rows.
- Report transaction timing and amount statistics.
- Show common fraud values for product, device, card, and email-domain columns.
- Profile numeric columns with missing counts and quantiles.
- Audit columns for missingness, constant values, duplicates, and likely identifiers.
- Analyze feature association with `isFraud`.
- Check missingness as a possible fraud signal.
- Estimate feature direction with logistic coefficients and LDA-style summaries.
- Sample fraud cases and compute pairwise similarity when fraud rows are available.
- Write Markdown and CSV outputs for inspection.

Example:

```bash
conda run -n fraud python fraud_exploration.py
```

Example outputs are kept under `outputs/example_exploration/`.

## 3. Feature Analysis And Selection

[fraud_feature_analysis.py](fraud_feature_analysis.py) adds a leakage-aware feature-analysis workflow.

It can:

- Split training data into train and held-out validation rows.
- Rank features using train-only signal against `isFraud`.
- Detect near-constant, identifier-like, weak, unstable, and redundant features.
- Compute feature stability across train and held-out rows.
- Compute pairwise redundancy and redundancy clusters.
- Estimate model-based feature importance.
- Compute class-specific permutation importance.
- Build a composite feature ranking.
- Select a practical feature subset.
- Save metrics, rankings, selected features, plots, and a feature-selection pipeline artifact.

Example:

```bash
conda run -n fraud python fraud_feature_analysis.py --data-dir data --output-dir outputs/full_feature_analysis
```

The repository also includes example feature-analysis outputs in `outputs/example_feature_analysis/`.

## 4. Model Training

[fraud_model_training.py](fraud_model_training.py) trains and compares fraud classifiers.

The current training workflow includes four model outputs:

- `rules_based`: a manually interpretable rule scorer built from high-risk thresholds and categories.
- `decision_tree`: a small human-readable decision tree that exports leaf rules.
- `neural_network`: a small NumPy neural network for binary fraud classification.
- `random_contender`: a seeded random baseline centered around the observed fraud rate.

The model training code:

- Uses a stratified validation split.
- Handles class imbalance with balanced sample weights where applicable.
- Uses selected features if a selected-feature CSV is provided.
- Otherwise ranks features from the current training split.
- Compares all model outputs with the same metrics.
- Writes confusion matrices for all model outputs.
- Writes ROC-AUC, PR-AUC, precision, recall, and F1.
- Saves model artifacts and the numeric preprocessor.
- Writes validation predictions.
- Writes Kaggle-format submission files with columns `TransactionID` and `isFraud`.

Default full-dataset run:

```bash
conda run -n fraud python fraud_model_training.py
```

This reads from `data/` and writes to `outputs/full_model_training/`.

Small smoke run:

```bash
conda run -n fraud python fraud_model_training.py \
  --max-rows 5000 \
  --max-features 12 \
  --epochs 10 \
  --hidden-units 6 \
  --max-rules 6 \
  --max-tree-depth 3
```

Expected model-training outputs include:

- `model_comparison_metrics.csv`
- `confusion_matrices.csv`
- `validation_predictions.csv`
- `validation_submission.csv`
- `test_submission.csv`
- `interpretable_rules.csv`
- `decision_tree_rules.csv`
- `decision_tree_rules.md`
- `rules_based_model.pkl`
- `decision_tree_model.pkl`
- `neural_network_model.pkl`
- `random_contender_model.pkl`
- `numeric_preprocessor.pkl`

## Data Files

The full Kaggle CSV files should live in `data/`:

- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

The repository also contains a tiny example subset under `data/example_subset/`. That subset is useful for checking loading and output paths, but it does not contain fraud-positive rows, so full model training requires the real training data.

## Runtime And Memory Notes

The full model-training script was profiled on capped row counts and then extrapolated to the full dataset.

Observed profiling:

- 50,000 train rows plus capped test rows, 25 epochs: about 19 seconds and 0.8 GB peak RAM.
- 100,000 train rows plus capped test rows, 25 epochs: about 44 seconds and 1.5 GB peak RAM.
- 50,000 train rows plus capped test rows, 250 epochs: about 24 seconds and 0.8 GB peak RAM.

Estimated full run:

- Full train rows: 590,540.
- Full test rows: 506,691.
- Expected runtime: about 6 to 8 minutes.
- Expected peak RAM: about 8 to 12 GB, with a conservative estimate below 16 GB.

The expected RAM usage should stay below a 32 GB cap.

## Tests

Tests are in [tests/test_fraud_exploration.py](tests/test_fraud_exploration.py).

They cover:

- Fraud fact-finding summaries.
- Dataset audit behavior.
- Binary feature-analysis behavior.
- Fraud similarity sampling and scoring.
- Feature ranking, redundancy, and selection.
- Model-training output shapes.
- Decision-tree readable rules.
- Submission formatting.
- Confusion matrix coverage for all model outputs.

Run tests with:

```bash
conda run -n fraud python -m pytest tests/test_fraud_exploration.py
```

## Dependencies

The project uses:

- `pandas`
- `numpy`
- `pytest` for tests

Runtime dependencies are listed in [requirements.txt](requirements.txt).

## Important Clarification

The current training script does not train a random forest. The four current outputs are rules-based, decision tree, neural network, and random-contender baseline. If the desired final model set is exactly neural network, random forest, and decision tree, the random-contender baseline should be replaced or supplemented with a real random forest implementation.
