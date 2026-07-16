# Requested Features To Do

## 1. Make The Data Machine Readable And Usable

- [x] Create a minimal example subset of the data to execute preprocessing code on.
- [x] Add shared IEEE-CIS train data loading that joins transaction and identity data on `TransactionID`.
- [x] Normalize identity column names between train and test files.
- [x] Cast known transaction and identity categorical columns to categorical dtypes.
- [x] Add a helper to load only confirmed fraud rows.
- [x] Add schema checks for required files and required columns.
- [x] Add explicit handling for missing transaction or identity rows after joins.
- [x] Add a reusable numeric encoder for mixed fraud-dataset columns.
- [x] Encode boolean-like values as numeric flags.
- [x] One-hot encode low-cardinality categoricals.
- [x] Frequency encode high-cardinality categoricals.
- [x] Add missing-value indicator columns for encoded features.
- [x] Impute numeric missing values with repeatable fitted values.
- [x] Persist fitted preprocessing artifacts for reuse on validation, test, and submission data.
- [x] Ensure train/test preprocessing uses the same fitted encodings and column order.
- [ ] Add project dependencies in `requirements.txt` only if new dependencies beyond pandas are introduced.

## 2. Make The Data Well Explored

- [x] Add descriptive fraud-only fact finding.
- [x] Summarize fraud case counts, transaction timing span, and transaction amount statistics.
- [x] Report top value distributions for fraud-case columns.
- [x] Profile numeric fraud-case columns with missing counts and quantiles.
- [x] Summarize fraud patterns by transaction time buckets.
- [x] Summarize common fraud-case value combinations.
- [x] Summarize fraud email-domain and card-type patterns.
- [x] Export the fraud fact-finder output as Markdown.
- [x] Add dataset audit outputs.
- [x] Validate binary targets and encode positive and negative classes.
- [x] Add univariate feature analysis for correlation with `isFraud`.
- [x] Add missingness-as-signal analysis for `isFraud`.
- [x] Compute feature correlations.
- [x] Compute standardized logistic-regression coefficients for association direction.
- [x] Add LDA-based class separability analysis.
- [x] Add optional SHAP attribution support.
- [x] Add cluster-first pairwise similarity analysis for fraud cases.
- [x] Sample fraud cases across clusters before computing pairwise similarity.
- [x] Compute similarity scores for numeric, boolean, and categorical comparison units.
- [x] Exclude missing comparison units from pairwise similarity aggregation.
- [x] Save similarity matrices, heatmaps, and summary statistics.
- [x] Add plots for exploration and binary feature-analysis outputs.
- [x] Add pytest coverage for fact finding, similarity analysis, and exploratory feature analysis.

## 3. Make The Data Efficient

- [x] Add a reusable binary feature-analysis package.
- [x] Add a leakage-safe general feature-analysis CLI.
- [x] Support YAML/JSON config files with CLI overrides.
- [x] Support excluded, id, and group columns.
- [x] Use train-only analysis before final held-out test evaluation.
- [x] Add feature stability analysis.
- [x] Add pairwise redundancy and redundancy-cluster analysis.
- [x] Add multivariate model-based feature importance.
- [x] Compute class-specific permutation importance.
- [x] Add composite feature ranking.
- [x] Add configurable feature-selection strategies.
- [x] Add feature-subset performance evaluation.
- [x] Define a minimum prediction-power threshold for dropping weak features.
- [x] Drop likely identifier, near-constant, redundant, unstable, and weak-contribution features.
- [x] Save selected features, metrics, final pipeline, plots, and reports.
- [x] Add a script that runs binary feature analysis on the IEEE-CIS fraud dataset.
- [x] Add pytest coverage for ranking, redundancy, selection, and feature-analysis pipelines.

## 4. Train Four Different Models

- [x] Train a rules-based fraud classifier on the selected feature set.
- [x] Define interpretable fraud rules from explored correlations, thresholds, and categorical patterns.
- [x] Train a completely human-understandable decision tree fraud classifier and export its rules.
- [x] Train a neural-network binary classifier for `isFraud`.
- [x] Add class-imbalance handling for model training.
- [x] Add validation split or cross-validation for all trained models.
- [x] Train a random-contender baseline model.
- [x] Compare all four models against common metrics.
- [x] Include confusion matrix, ROC-AUC, precision, recall, F1, and PR-AUC.
- [x] Save model artifacts and preprocessing artifacts for each model.
- [x] Save predictions in Kaggle submission format.
- [x] Add repeatable training scripts for all four models.
- [x] Add tests for model input shape, prediction output shape, and submission formatting.
