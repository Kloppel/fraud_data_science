# Workflow Overview

Built incrementally, one function at a time, while walking through the fraud-detection
pipeline. Order follows the actual data flow:

`download_data.py` → `fraud_preprocessing.py` → `fraud_exploration.py` →
`fraud_feature_analysis.py` → `fraud_model_training.py` → `fraud_model_context.py` →
`fraud_results_plots.py`

## download_data.py

- **(script body, no functions)** — calls `kagglehub.competition_download('ieee-fraud-detection')`
  to fetch the raw Kaggle CSVs to a local cache path and prints that path. Output is meant to be
  moved into `data/` for the rest of the pipeline to read.

## fraud_preprocessing.py

Shared data-loading and numeric-encoding code that every later stage builds on.

- **`FraudDataLoader.__init__`** — stores the `data_dir` (defaults to `data/`) to read CSVs from.
- **`FraudDataLoader.load_train` / `load_test`** — read the transaction + identity CSVs for a
  split and left-join them on `TransactionID` (`load_train` requires `isFraud` to be present,
  `load_test` does not, since test has no labels).
- **`FraudDataLoader.load_fraud_only`** — convenience wrapper: loads train, then filters to
  `isFraud == 1` for fraud-focused analysis.
- **`FraudDataLoader._join_transaction_identity`** — does the actual merge: normalizes identity
  column names, checks required columns exist, left-joins on `TransactionID`, and adds a
  `has_identity` flag (1 if a row had a matching identity record, 0 if not — since not every
  transaction has one).
- **`FraudDataLoader.normalize_identity_columns`** (static) — renames `id-01`...`id-38` (test-file
  hyphen style) to `id_01`...`id_38` (train-file underscore style) so both splits line up.
- **`FraudDataLoader._require_files`** — raises `FileNotFoundError` if any expected CSV is missing
  from `data_dir`.
- **`FraudDataLoader._require_columns`** (static) — raises `ValueError` if required columns
  (`TransactionID`, `isFraud`) aren't present in a loaded frame.
- **`FraudDataLoader._cast_known_categoricals`** (static) — casts the known categorical columns
  (`ProductCD`, `card1`-`card6`, `addr1/2`, email domains, `M1`-`M9`, `DeviceType`, `DeviceInfo`,
  `id_12`-`id_38`) to pandas `category` dtype so downstream code treats them as categorical, not
  numeric.
- **`NumericPreprocessor.__init__`** — sets up an empty fitted-state container: which columns to
  skip (`TransactionID`, `isFraud`), and a cardinality threshold (default 30) for choosing
  one-hot vs. frequency encoding.
- **`NumericPreprocessor.fit`** — for every non-excluded column, decides an encoding "plan" and
  remembers it: boolean-like (`T`/`F`) → boolean plan; numeric → store the median for later
  imputation; low-cardinality categorical → one-hot plan with the fixed category list;
  high-cardinality categorical → frequency-encoding plan (map each category to how often it
  appears). Then computes `output_columns_` once so all future `transform` calls produce the same
  column set/order.
- **`NumericPreprocessor.transform`** — applies the fitted plan to any dataframe: adds
  `__was_missing` indicator columns, maps booleans to 0/1, imputes numeric NaNs with the fitted
  median, expands one-hot columns, or maps categories to their fitted frequency. Reindexes to the
  exact `output_columns_` so train/val/test always match shape.
- **`NumericPreprocessor.fit_transform`** — convenience: `fit` then `transform` on the same data.
- **`NumericPreprocessor.save` / `load`** — pickle the fitted preprocessor to disk and reload it,
  so the exact same encoding can be reapplied later (e.g., to test data at inference time).
- **`NumericPreprocessor._as_category_strings`** (static) — converts a series to strings, filling
  missing values with a `__missing__` sentinel token before category comparisons.
- **`NumericPreprocessor._is_boolean_like`** (static) — checks whether a column's non-missing
  values are a subset of `{T, F, True, False}`, to decide if it should use the boolean plan.
- **`main`** — CLI entry point: loads a data dir (defaults to the small `data/example_subset/`),
  fits and applies the preprocessor, and writes `numeric_train.csv`, `row_metadata.csv`, and the
  pickled `numeric_preprocessor.pkl` to an output dir (defaults to `outputs/example_preprocessing/`).

## fraud_exploration.py

Descriptive/exploratory analysis of fraud vs. non-fraud rows. Imports `FraudDataLoader` from
`fraud_preprocessing.py`, so it builds directly on stage 1.

**Shared helpers**
- **`validate_binary_target`** — checks a target column only contains `{0, 1}` (post-mapping from
  arbitrary positive/negative labels) and raises if not; returns it as clean 0/1 ints.
- **`infer_feature_type`** — classifies any column as `"boolean"` (values subset of
  `T/F/True/False/0/1`), `"numeric"`, or `"categorical"`, used throughout the file to branch logic.

**`FraudFactFinder`** — descriptive stats scoped to confirmed-fraud rows only.
- **`__init__`** — stores a copy of the fraud-only dataframe.
- **`summary`** — headline numbers: fraud case count, the day-span of `TransactionDT` (converted
  from seconds), and `TransactionAmt` total/mean/median. Returns all-NaN placeholders if no fraud
  rows exist.
- **`distribution`** — top-N value counts (normalized to fractions by default) for one column.
- **`numeric_profile`** — count/missing/mean/median/std/min/max plus 5/25/75/95th percentiles for
  a numeric column.
- **`time_bucket_summary`** — buckets `TransactionDT` seconds-of-day into N bins (default 24, i.e.
  hourly) and counts fraud rows per bucket — a proxy "time of day" fraud histogram.
- **`common_combinations`** — group-by on a set of columns (e.g. `ProductCD` + `DeviceType`),
  returns the top-N most frequent combinations with their share of fraud rows.
- **`email_domain_summary`** — compares `P_emaildomain` vs `R_emaildomain`: top values for each,
  plus match/differ/either-missing rates.
- **`card_type_summary`** — a `card4` × `card6` cross-tab with margins, for fraud rows.
- **`to_markdown`** — assembles all of the above into one Markdown report string.
- **`to_report`** — writes `to_markdown()`'s output to a file, creating parent dirs as needed.

**Dataset-wide audit and association**
- **`audit_dataset`** — one row per feature with quality flags: missingness, unique count,
  variance, "is constant," "is near-zero-variance," "is likely an identifier" (near-100% unique
  discrete column), "is a duplicate of another column," and missingness split by fraud class.
- **`analyze_univariate_features`** — for every feature, computes association with `isFraud`
  (numeric → `_numeric_association`, categorical → `_categorical_association`) and ranks by
  association strength.
- **`analyze_missingness_signal`** — checks whether a column's *missingness itself* (not its
  value) differs between fraud and non-fraud rows — i.e., "is NaN-ness predictive."
- **`numeric_feature_correlations`** — flags numeric column pairs whose Pearson correlation
  exceeds a threshold (default 0.8), useful for spotting redundant features.
- **`standardized_logistic_coefficients`** — fits a standardized logistic regression (via
  sklearn if available, else falls back to per-feature point-biserial correlation) and reports
  which class each feature/level favors.
- **`lda_separability_summary`** — fits a one-component LDA (or a manual pooled-variance fallback)
  to get per-feature class-separation loadings.
- **`shap_status`** — checks whether the `shap` package is installed; this repo doesn't require it
  but leaves the hook open for later.

**Fraud-case similarity**
- **`sample_fraud_cases_for_similarity`** — stratified sample of fraud rows, allocating sample
  slots proportionally across a cluster/category column (default `ProductCD`) so the sample isn't
  dominated by one product type.
- **`pairwise_similarity`** — builds an N×N similarity matrix between sampled fraud rows: numeric
  columns score by closeness relative to a percentile-based distance cutoff, categorical columns
  score 1/0 for exact match; missing values are excluded from a pair's average rather than
  penalized.
- **`similarity_summary`** — mean/median/min/max similarity across all off-diagonal pairs, plus a
  count of NaN pairs.

**Orchestration and output writers**
- **`run_exploration`** — the actual pipeline: loads train data, runs `FraudFactFinder` and
  `audit_dataset` unconditionally; if both fraud classes are present, also runs the univariate/
  missingness/correlation/coefficient/LDA analyses and writes their CSVs plus two SVG bar charts;
  if at least 2 fraud rows exist, also runs the similarity sampling + matrix + heatmap. Writes a
  `run_summary.json` listing every output file produced.
- **`write_bar_svg`** — hand-rolled SVG horizontal bar chart (no matplotlib dependency) for a
  label/value dataframe.
- **`write_heatmap_svg`** — hand-rolled SVG grid heatmap for the similarity matrix, shading each
  cell by value (NaN cells rendered as a fixed neutral shade).

**Private helpers** (prefixed `_`, not part of the public API)
- **`_find_duplicate_columns`** — pairwise `.equals()` check to map each column to the first
  earlier column it's identical to.
- **`_numeric_series`** — coerces a column to numeric, mapping boolean-like values to 0/1 first.
- **`_class_missingness`** — missingness rate of a column restricted to one target class.
- **`_numeric_association` / `_categorical_association`** — the per-feature-type association
  metrics (means by class, point-biserial correlation or target-rate range, and a signed AUC-based
  "association_strength") used by `analyze_univariate_features` and `analyze_missingness_signal`.
- **`_signed_auc_from_scores`** — rank-based AUC computation (Mann-Whitney U formulation) rescaled
  to `[-1, 1]` so positive means "high values favor fraud."
- **`_safe_corr`** — Pearson correlation that returns NaN instead of erroring on too-few-rows or
  no-variance edge cases.
- **`_standardize`** — z-score a series (0 if std is 0).
- **`_filled_category`** — casts to string, filling NaN with the `__missing__` token.
- **`_numeric_distance_cutoffs`** — for each numeric similarity column, computes a percentile of
  all pairwise absolute differences, used as the "maximum distance" denominator in similarity
  scoring.
- **`_default_similarity_columns`** — the fallback column list for `pairwise_similarity` when none
  is given (amount, product, card, email, device, `M1`-`M3`), or the first 20 non-ID columns if
  none of those are present.
- **`_model_matrix`** — builds a standardized numeric + one-hot design matrix (capping category
  levels, collapsing rare ones into `__other__`) for the logistic regression / LDA fits.
- **`_float_or_nan` / `_float_or_zero`** — null-safe float coercion helpers.
- **`_dict_markdown` / `_series_markdown` / `_frame_markdown`** — render a dict/Series/DataFrame
  as a Markdown section for `FraudFactFinder.to_markdown`.
- **`_svg_escape`** — escapes `&`/`<`/`>` for safe SVG text embedding.
- **`_markdown_table` / `_markdown_cell`** — generic DataFrame-to-Markdown-table rendering, with
  `|` escaped in cell values.
- **`main`** — CLI entry point: parses `--data-dir`/`--output-dir`/`--similarity-sample-size`,
  calls `run_exploration`, and prints the JSON result summary.

## fraud_feature_analysis.py

Leakage-aware feature ranking and selection: everything that decides feature signal is computed
on a train-only split, then validated against a held-out split, so no held-out information leaks
into the ranking.

**Config**
- **`FeatureAnalysisConfig`** (dataclass) — every tunable knob: data/output dirs, target/id/group/
  excluded columns, held-out fraction, random seed, row cap, top-N, and thresholds for prediction
  power / redundancy / stability / near-constant. `from_mapping` builds one from a dict (splitting
  comma-separated strings into lists for column-list fields); `to_dict` serializes it back for the
  saved run summary.
- **`load_config`** — reads a JSON or YAML config file; falls back to a hand-rolled
  `_parse_simple_yaml` if `PyYAML` isn't installed, so YAML configs work without a hard dependency.
- **`merge_config`** — loads a config file, then overlays any non-`None` CLI overrides on top, and
  builds the final `FeatureAnalysisConfig`.

**Orchestration**
- **`run_feature_analysis`** — the main pipeline: loads train data, does a `train_heldout_split`,
  picks feature columns, then — only if both fraud classes are present — runs the full ranking/
  stability/redundancy/importance/permutation/composite/selection/evaluation chain (below);
  otherwise writes empty placeholder tables and a "skipped" report. Writes every intermediate CSV,
  the Markdown report, an SVG bar chart of the composite ranking, a pickled
  `feature_selection_pipeline.pkl` (config + selected feature list), and `run_summary.json`.
- **`select_input_columns`** — feature columns = all columns minus id/group/excluded/target.
- **`validate_binary_target`** / **`has_two_classes`** — same style of target-label checks as in
  `fraud_exploration.py`, duplicated locally so this module has no cross-module runtime coupling
  beyond `FraudDataLoader`.
- **`train_heldout_split`** — deterministic split (stratified per-class sampling when both classes
  are present) so evaluation rows are held out from every ranking computation.

**Per-feature analysis (train-only)**
- **`audit_features`** — per-feature type/missingness/uniqueness/near-constant/identifier-like
  flags (same idea as `fraud_exploration.audit_dataset`, but scoped to the selected feature list).
- **`rank_features`** — ranks features by `feature_score` (signed AUC) computed only on the train
  split.
- **`feature_stability`** — compares each feature's signed-AUC score on train vs. held-out data;
  flags a feature `stable` if the absolute difference is under the configured threshold — catches
  features whose apparent signal doesn't generalize.
- **`redundancy_analysis`** — builds a model matrix, correlates all encoded columns, and returns
  pairs above the redundancy threshold plus, via `redundancy_clusters`, connected-component groups
  of mutually redundant features.
- **`redundancy_clusters`** — union-find-style graph traversal (BFS/DFS via a stack) over the
  redundant pairs to group them into clusters.
- **`model_based_importance`** — fits standardized logistic regression (falls back to the
  train-only AUC ranking if sklearn errors out) and takes `abs(coefficient)` per encoded column,
  rolled back up to the root feature via `root_feature`.
- **`class_specific_permutation_importance`** — for each feature and each class label, shuffles
  that feature's values within rows of that class on the held-out (or train, if held-out lacks two
  classes) split and measures how much the signed-AUC score drops — a class-aware permutation
  importance.
- **`composite_ranking`** — merges audit + train ranking + stability + model importance +
  permutation importance into one table, normalizes each importance column to `[0, 1]`, and
  combines them into a weighted `composite_score` (45% prediction power, 35% model importance, 20%
  permutation importance) minus a penalty for being constant/near-constant/identifier-like/unstable.
- **`select_features`** — walks the composite ranking and marks each feature `selected` unless it
  trips one of: likely-identifier, near-constant, redundant (via `redundant_features_to_drop`),
  unstable (AUC delta over threshold), or weak prediction power — records the specific
  `drop_reason`(s).
- **`evaluate_feature_subsets`** — evaluates named subset "strategies" (`top_10`, `balanced`,
  `conservative`) via `subset_auc` on the held-out split, to show how selection choices trade off.
- **`subset_auc`** — scores a held-out split using train-fitted per-feature target rates averaged
  across a feature subset, then converts the resulting signed AUC to a plain `[0, 1]` AUC.
- **`feature_report`** — renders the top composite features and subset performance table into a
  Markdown report string.

**Shared low-level helpers** (near-duplicates of ones in `fraud_exploration.py`, kept local so this
module stays self-contained)
- **`infer_feature_type`** — boolean/numeric/categorical classification.
- **`feature_score`** — computes a signed AUC of one feature against the target (numeric passes
  through, boolean maps to 0/1, categorical uses target-rate-encoding first).
- **`signed_auc`** — rank-based AUC rescaled to `[-1, 1]`.
- **`model_matrix`** — standardized numeric/boolean columns + capped one-hot categoricals, used by
  `redundancy_analysis` and `model_based_importance`.
- **`root_feature`** — strips the `__category` suffix off an encoded one-hot column name to get
  back the original feature name.
- **`redundant_features_to_drop`** — for each redundant pair, keeps the higher-composite-score
  feature and marks the other for dropping.
- **`fill_feature`** — fills NaN with the `__missing__` token before treating a column as string
  categories.
- **`standardize`** — z-score a numeric series.
- **`_nan_to_zero`** — null-safe float coercion to 0.0.
- **`_parse_simple_yaml`** — minimal `key: value` / `key: [a, b]` YAML-subset parser used when
  `PyYAML` isn't installed.

**Output writers / CLI**
- **`write_bar_svg`** — hand-rolled SVG bar chart of the composite ranking (same style as in
  `fraud_exploration.py`, but keyed to `composite_score`).
- **`markdown_table` / `markdown_cell` / `svg_escape`** — same generic Markdown/SVG rendering
  helpers as `fraud_exploration.py`, duplicated locally.
- **`empty_ranking` / `empty_stability` / `empty_model_importance` / `empty_permutation` /
  `empty_composite`** — typed empty-DataFrame placeholders used when a train split has only one
  target class (so downstream CSV writes still succeed with the right columns).
- **`parse_args`** — CLI flags mirroring `FeatureAnalysisConfig`'s fields, plus `--config` for a
  JSON/YAML file.
- **`main`** — parses args, merges them over an optional config file, runs
  `run_feature_analysis`, and prints the JSON summary.

## fraud_model_training.py

Trains and compares four fraud classifiers, all sharing the same validation split and metrics.

- **`ModelTrainingConfig`** (dataclass) — knobs for data/output dirs, an optional path to a
  selected-features CSV (from stage 3), split fraction/seed, row/feature caps, and hyperparameters
  for the rule count, tree depth, and neural net (hidden units, learning rate, epochs).

**`RuleBasedFraudClassifier`** — an interpretable scorer built from simple high-precision rules.
- **`fit`** — for each feature, generates numeric threshold rules (`_numeric_rules`) or categorical
  equality rules (`_categorical_rules`), keeps the top `max_rules` sorted by precision/lift/support.
- **`predict_proba`** — starts every row at the base fraud rate, then for each matching rule adds a
  precision-weighted nudge toward that rule's precision, normalizing by total rule weight matched.
- **`save`** — pickles the fitted classifier.
- **`rules_frame`** — the fitted rules as a DataFrame (for the `interpretable_rules.csv` output).
- **`_numeric_rules`** — proposes `>=`/`<=` threshold rules at the 25/50/75th percentiles, oriented
  toward whichever direction fraud cases skew, and keeps only ones passing `_build_rule`'s
  precision/lift bar.
- **`_categorical_rules`** — proposes `==value` rules for categories with support ≥ `min_support`
  whose fraud rate beats the base rate by at least 25% lift.
- **`_build_rule`** — shared threshold-rule builder: computes support/precision/lift for a
  candidate mask, rejects if support or lift is too low.
- **`_rule_mask`** (static) — evaluates one fitted rule against a dataframe to get a boolean mask.

**`HumanDecisionTreeFraudClassifier`** — a small, from-scratch decision tree with plain-language
leaf paths.
- **`fit`** — computes balanced sample weights, then recursively builds the tree via `_build_node`.
- **`predict_proba`** — walks each row down the tree via `_predict_row`.
- **`rules_frame`** / **`rules_markdown`** — flattens the tree into leaf paths
  (`feature <= x AND feature2 == y`) with each leaf's predicted probability, sample count, and
  training fraud rate; `rules_markdown` renders that as a readable Markdown report.
- **`save`** — pickles the fitted tree.
- **`_build_node`** — recursive node builder: stops at `max_depth`, too few samples, or a pure
  class; otherwise finds `_best_split` and recurses into left/right children.
- **`_best_split`** — tries every candidate split (from `_split_candidates`) across every feature,
  picks the one with the highest weighted-Gini gain over the parent node.
- **`_split_candidates`** — numeric features get `<=` thresholds at the 25/50/75th percentiles;
  categorical features get `==` candidates for their most common values (capped at
  `max_category_values`).
- **`_candidate_mask`** (static) — evaluates one split candidate into a boolean mask.
- **`_leaf_node`** (static) — a leaf's prediction is its weighted fraud rate.
- **`_predict_row`** — recursively follows the split condition at each node down to a leaf.
- **`_collect_rules`** — recursive traversal building the `path`/`prediction`/`samples`/
  `fraud_rate` rows used by `rules_frame`, formatting each split with `format_tree_condition`.

**`NeuralNetworkFraudClassifier`** — a minimal one-hidden-layer network implemented directly in
NumPy (no ML framework dependency).
- **`fit`** — initializes small random weights, then runs plain gradient descent for `epochs`
  steps: forward pass (`tanh` hidden layer, sigmoid output), weighted error (via
  `balanced_sample_weights` so the minority fraud class isn't drowned out), backprop gradients,
  and a manual weight update — no autodiff library involved.
- **`predict_proba`** — forward pass only.
- **`save`** — pickles the fitted weights.

**`RandomContenderClassifier`** — a seeded random baseline, useful as a sanity-check floor.
- **`fit`** — remembers the observed fraud base rate.
- **`predict_proba`** — draws Beta(2,5) noise and blends it 70/30 with the base rate, so predictions
  cluster around the true prior but aren't literally identical every row.
- **`save`** — pickles it.

**Orchestration**
- **`run_model_training`** — loads training data, validates both classes are present, does a
  `stratified_split`, picks features via `choose_features`, then calls `train_model_bundle` and
  `write_test_submission_if_available`; writes `run_summary.json`.
- **`train_model_bundle`** — fits the shared `NumericPreprocessor` (for the neural net's numeric
  input) plus all four classifiers on the same train split, scores all four on the same validation
  split into one `predictions` frame, computes comparison metrics via `compare_models`, and writes
  every CSV/Markdown/pickle output (metrics, confusion matrices, validation predictions/submission,
  rule tables, and all four model pickles plus the preprocessor pickle).
- **`write_test_submission_if_available`** — if `data/test_transaction.csv` etc. exist and contain
  all the chosen features, scores the (neural network only) test set and writes
  `test_submission.csv`; returns `False` silently if test data isn't available so the smoke-test
  workflow doesn't require it.
- **`choose_features`** — uses the stage-3 selected-features CSV if given and non-empty; otherwise
  ranks all candidate columns by `quick_feature_power` and takes the top `max_features`.
- **`load_selected_features`** — reads a selected-features CSV, filtering to rows where
  `selected` is true if that column exists.
- **`compare_models`** — for every model column in the predictions frame, thresholds at 0.5 and
  computes ROC-AUC, PR-AUC, precision, recall, and F1, plus confusion counts — same metrics for
  every model side by side.
- **`write_submission`** — renames a model's score column to `isFraud` and writes the Kaggle-format
  `TransactionID,isFraud` CSV.
- **`stratified_split`** — per-class sampling (like `train_heldout_split` in stage 3) to build a
  validation split with both classes represented; raises if a class has under 2 rows.
- **`validate_two_class_target`** — raises unless the target is exactly `{0, 1}` (training requires
  both classes, unlike the exploration/feature-analysis stages which tolerate one-class data).
- **`quick_feature_power`** — absolute deviation of a feature's ROC-AUC from 0.5 (used only for the
  fallback feature-choice ranking, distinct from stage 3's more thorough ranking).

**Shared metric/utility helpers**
- **`roc_auc_score`** — rank-based ROC-AUC computed manually (Mann-Whitney U formulation), no
  sklearn dependency.
- **`average_precision_score`** — manual precision-recall AUC via a running precision-at-each-
  true-positive sum, divided by total positives.
- **`confusion_counts`** — tn/fp/fn/tp counts at a fixed 0.5 threshold.
- **`f1_score`** — computed from `confusion_counts` via `safe_divide`.
- **`safe_divide`** — returns 0.0 instead of raising/NaN on divide-by-zero.
- **`balanced_sample_weights`** — per-row weight inversely proportional to that row's class
  frequency, so the tree and neural net don't just learn to predict "not fraud" everywhere.
- **`weighted_gini`** — weighted Gini impurity, used by the decision tree's split search.
- **`format_tree_condition`** — renders one split as human-readable text (`<=`/`>` for numeric,
  `==`/`!=` for categorical), depending on which branch (`positive`) is being described.
- **`sigmoid`** — numerically clipped logistic sigmoid.
- **`fill_category`** — fills NaN with `__missing__` before string comparison (same pattern as the
  other stages).
- **`parse_args`** — CLI flags mirroring `ModelTrainingConfig`.
- **`main`** — builds a config from CLI args, calls `run_model_training`, prints the JSON summary.

## fraud_model_context.py

Turns the raw model-training artifacts (stage 4's CSVs/pickles) into human-readable "why did the
model decide this" reports. Imports `fraud_model_training` directly (not just its output files) so
it can unpickle and walk the decision tree's internal structure.

- **`run_model_context`** — reads stage 4's metrics/confusion/predictions CSVs and run summary,
  builds a training-performance context table, a decision-tree impact table, and a combined
  Markdown report; writes all three plus a `model_context_summary.json`.
- **`summarize_training`** — turns raw metrics into narrated rows: which model had the best
  ROC-AUC/PR-AUC/precision/recall/F1 (with a plain-English `context` sentence for each), the
  validation set's fraud-positive rate, echoed run-summary stats (row counts, feature count), and
  a one-line confusion-matrix summary per model.
- **`decision_tree_decision_impacts`** — unpickles the trained `HumanDecisionTreeFraudClassifier`
  (via `load_pickle_with_main_mapping`) and walks its tree via `collect_decision_impacts`, then
  sorts by the magnitude of each split's effect on predicted fraud probability.
- **`collect_decision_impacts`** — recursive tree walk: at each internal node, computes how much
  each branch (left/right) shifts the predicted probability relative to the parent
  (`child_prediction - parent_prediction`), recording the path, the branch condition (via
  `fraud_model_training.format_tree_condition`), and the child's sample count/fraud rate.
- **`context_markdown`** — assembles the full report: run summary, model performance table,
  confusion-matrix bullets, validation prediction stats, the decision tree's leaf rules, and its
  decision-impact table.
- **`MainMappingUnpickler`** — a custom `pickle.Unpickler` subclass that redirects classes pickled
  under `__main__` (i.e., when `fraud_model_training.py` was run directly as a script rather than
  imported as a module) to their real location in the `fraud_model_training` module — otherwise
  unpickling the saved tree/model files would fail with a "no module __main__.ClassName" error.
- **`load_pickle_with_main_mapping`** — reads bytes and unpickles them through that custom
  unpickler.
- **`read_csv` / `read_json`** — return an empty DataFrame/dict instead of raising if the expected
  stage-4 artifact file doesn't exist, so this stage degrades gracefully if training wasn't run.
- **`markdown_table` / `markdown_cell`** — the same generic table renderer pattern as earlier
  stages, with one addition: floats are formatted to 6 significant digits (`{value:.6g}`) instead
  of raw `str()`.
- **`parse_args`** — `--model-dir` / `--output-dir` CLI flags.
- **`main`** — calls `run_model_context` and prints the JSON summary.

## fraud_results_plots.py

The final stage: reads CSV/JSON artifacts from exploration, feature analysis, and model training,
and renders them as dependency-free SVG charts (no matplotlib) plus an index page. Purely a
presentation layer — it doesn't recompute anything, just visualizes existing outputs.

- **`run_results_plots`** — checks for each expected upstream artifact and, if present, renders the
  corresponding chart: model metrics → grouped bar chart, confusion matrices → a TN/FP/FN/TP grid,
  decision-tree leaf predictions → horizontal bars, feature-selection counts → horizontal bars,
  top composite feature scores → horizontal bars, top-missing-value features → horizontal bars.
  Writes `plot_summary.json` (which plots were produced) and `plot_index.md` (a linked index).
  Any missing input file is skipped rather than erroring, so this stage works with partial pipeline
  runs.
- **`read_csv`** — returns an empty DataFrame if the file doesn't exist (same graceful-degradation
  pattern as `fraud_model_context.py`).
- **`write_grouped_metric_bars`** — one grouped bar cluster per model, one bar per metric
  (`roc_auc`, `pr_auc`, `precision`, `recall`, `f1`), each metric given a distinct color.
- **`write_confusion_grid`** — a 2×2 TN/FP/FN/TP grid per model, shading each cell's color
  intensity by its count relative to that model's max cell.
- **`write_horizontal_bars`** — generic top-N horizontal bar chart used for four different plots
  (leaf predictions, feature-selection counts, composite scores, missing-value fractions); sorts
  descending by value and truncates long labels.
- **`plot_index_markdown`** — a Markdown list linking every SVG artifact that was actually written.
- **`svg_header`** — the shared `<svg>`/background/title boilerplate reused by every chart writer.
- **`rect`** — one `<rect>` element string.
- **`text`** — one `<text>` element string, with size/color/weight options.
- **`shade`** — blends a base color toward white by `1 - intensity`, used to color confusion-grid
  cells by relative magnitude (clamped so even zero-count cells stay visibly tinted).
- **`numeric`** — null/type-safe float coercion, defaulting to 0.0.
- **`escape`** — the same `&`/`<`/`>` SVG-text escaping helper seen in earlier stages.
- **`parse_args`** — CLI flags for the four input directories, output directory, and `--top-n`.
- **`main`** — calls `run_results_plots` and prints the JSON summary.
