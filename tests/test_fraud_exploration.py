import math

import pandas as pd
import pytest

from fraud_feature_analysis import (
    FeatureAnalysisConfig,
    audit_features as audit_feature_efficiency,
    composite_ranking,
    feature_stability,
    rank_features,
    redundancy_analysis,
    run_feature_analysis,
    select_features,
)
from fraud_model_training import (
    ModelTrainingConfig,
    NeuralNetworkFraudClassifier,
    compare_models,
    run_model_training,
    write_submission,
)
from fraud_exploration import (
    SECONDS_PER_DAY,
    FraudFactFinder,
    analyze_missingness_signal,
    analyze_univariate_features,
    audit_dataset,
    lda_separability_summary,
    pairwise_similarity,
    run_exploration,
    sample_fraud_cases_for_similarity,
    similarity_summary,
    standardized_logistic_coefficients,
    validate_binary_target,
)


def test_fact_finder_summarizes_fraud_cases_and_writes_markdown(tmp_path):
    df = pd.DataFrame(
        {
            "TransactionDT": [0, SECONDS_PER_DAY, 3 * SECONDS_PER_DAY],
            "TransactionAmt": [10.0, 20.0, 30.0],
            "ProductCD": ["W", "W", "C"],
            "DeviceType": ["mobile", "mobile", "desktop"],
            "card4": ["visa", "visa", "mastercard"],
            "card6": ["credit", "credit", "debit"],
            "P_emaildomain": ["gmail.com", "gmail.com", None],
            "R_emaildomain": ["gmail.com", "yahoo.com", None],
        }
    )

    finder = FraudFactFinder(df)
    summary = finder.summary()

    assert summary["n_fraud_cases"] == 3
    assert summary["transaction_dt_days_span"] == pytest.approx(3.0)
    assert summary["transaction_amt_median"] == pytest.approx(20.0)
    assert finder.distribution("ProductCD", normalize=False)["W"] == 2
    assert finder.email_domain_summary()["match_pct"] == pytest.approx(1 / 3)

    report = tmp_path / "fraud_report.md"
    finder.to_report(report)
    assert "# Fraud Fact Finder Report" in report.read_text()


def test_audit_flags_missingness_duplicate_identifier_and_constant_columns():
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "isFraud": [0, 0, 1, 1],
            "id_like": [10, 11, 12, 13],
            "constant_col": [5, 5, 5, 5],
            "amount": [1.0, 2.0, 10.0, 12.0],
            "amount_copy": [1.0, 2.0, 10.0, 12.0],
            "missing_signal": [None, "x", None, None],
        }
    )

    audit = audit_dataset(df).set_index("feature")

    assert audit.loc["id_like", "is_likely_identifier"]
    assert audit.loc["constant_col", "is_constant"]
    assert audit.loc["amount_copy", "is_duplicate"]
    assert audit.loc["amount_copy", "duplicate_of"] == "amount"
    assert audit.loc["missing_signal", "missingness_class_1"] > audit.loc["missing_signal", "missingness_class_0"]


def test_binary_feature_analysis_reports_expected_association_direction():
    df = pd.DataFrame(
        {
            "TransactionID": range(8),
            "isFraud": [0, 0, 0, 0, 1, 1, 1, 1],
            "amount": [1, 2, 1, 2, 10, 12, 11, 13],
            "inverse": [10, 9, 11, 12, 2, 1, 3, 2],
            "domain": ["a", "a", "b", "b", "x", "x", "x", "a"],
            "missing": [1, 2, 3, 4, None, None, None, 8],
        }
    )

    target = validate_binary_target(df["isFraud"])
    univariate = analyze_univariate_features(df).set_index("feature")
    missingness = analyze_missingness_signal(df).set_index("feature")
    coeffs = standardized_logistic_coefficients(df)
    lda = lda_separability_summary(df).set_index("feature")

    assert target.tolist() == df["isFraud"].tolist()
    assert univariate.loc["amount", "signed_auc"] > 0
    assert univariate.loc["inverse", "signed_auc"] < 0
    assert missingness.loc["missing", "missing_rate_diff_class_1_minus_0"] > 0
    assert coeffs.iloc[0]["abs_coefficient"] >= coeffs.iloc[-1]["abs_coefficient"]
    assert lda.loc["amount", "lda_loading"] > 0


def test_validate_binary_target_rejects_single_class_subset():
    with pytest.raises(ValueError):
        validate_binary_target(pd.Series([0, 0, 0]))


def test_similarity_samples_by_cluster_and_scores_pairs():
    fraud = pd.DataFrame(
        {
            "TransactionID": [10, 11, 12, 13],
            "ProductCD": ["W", "W", "C", "C"],
            "TransactionAmt": [10.0, 10.0, 200.0, 220.0],
            "card4": ["visa", "visa", "mastercard", "mastercard"],
            "M1": ["T", "T", "F", None],
        }
    )

    sampled = sample_fraud_cases_for_similarity(fraud, sample_size=4, random_state=0)
    matrix = pairwise_similarity(sampled)
    stats = similarity_summary(matrix)

    assert matrix.shape == (4, 4)
    assert matrix.loc["10", "11"] == pytest.approx(1.0)
    assert matrix.loc["10", "12"] < matrix.loc["10", "11"]
    assert stats["n_sampled"] == 4
    assert not math.isnan(stats["mean_similarity"])


def test_run_exploration_on_minimal_subset_writes_skip_outputs(tmp_path):
    result = run_exploration(
        data_dir="data/example_subset",
        output_dir=tmp_path,
        similarity_sample_size=5,
    )

    assert result["rows"] == 40
    assert result["fraud_rows"] == 0
    assert (tmp_path / "fraud_fact_finder.md").exists()
    assert (tmp_path / "dataset_audit.csv").exists()
    assert (tmp_path / "binary_feature_analysis_skipped.txt").exists()
    assert (tmp_path / "fraud_similarity_skipped.txt").exists()


def test_feature_analysis_ranks_redundancy_and_selection_pipeline():
    df = pd.DataFrame(
        {
            "TransactionID": range(12),
            "isFraud": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "amount": [1, 2, 1, 2, 3, 2, 20, 21, 19, 22, 20, 23],
            "amount_copy": [1, 2, 1, 2, 3, 2, 20, 21, 19, 22, 20, 23],
            "constant": ["x"] * 12,
            "domain": ["a", "a", "b", "b", "a", "b", "z", "z", "z", "y", "z", "y"],
        }
    )
    train = df.iloc[:10].reset_index(drop=True)
    heldout = df.iloc[10:].reset_index(drop=True)
    features = ["amount", "amount_copy", "constant", "domain"]

    audit = audit_feature_efficiency(train, features)
    ranking = rank_features(train, features)
    stability = feature_stability(train, heldout, features)
    redundancy_pairs, _ = redundancy_analysis(train, features, threshold=0.99)
    composite = composite_ranking(
        audit,
        ranking,
        stability,
        pd.DataFrame({"feature": ["amount", "amount_copy"], "importance": [1.0, 0.9]}),
        pd.DataFrame(
            {
                "feature": ["amount", "amount_copy"],
                "class_label": [1, 1],
                "permutation_importance": [0.2, 0.2],
            }
        ),
    )
    selected = select_features(
        composite,
        audit,
        redundancy_pairs,
        min_prediction_power=0.01,
        stability_threshold=1.0,
    ).set_index("feature")

    assert ranking.set_index("feature").loc["amount", "prediction_power"] > 0
    assert not redundancy_pairs.empty
    assert selected.loc["constant", "selected"] == False
    assert selected.loc["amount", "selected"] != selected.loc["amount_copy", "selected"]


def test_feature_analysis_cli_workflow_runs_on_minimal_subset(tmp_path):
    result = run_feature_analysis(
        FeatureAnalysisConfig(
            data_dir="data/example_subset",
            output_dir=str(tmp_path),
            max_rows=20,
        )
    )

    assert result["rows"] == 20
    assert result["has_two_classes"] is False
    assert (tmp_path / "feature_audit.csv").exists()
    assert (tmp_path / "selected_features.csv").exists()
    assert (tmp_path / "feature_selection_pipeline.pkl").exists()


def test_model_training_shapes_metrics_and_submission_formatting(tmp_path):
    train_transaction = pd.DataFrame(
        {
            "TransactionID": range(1, 13),
            "isFraud": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "TransactionDT": range(12),
            "TransactionAmt": [8, 12, 9, 11, 13, 10, 90, 95, 88, 100, 93, 97],
            "ProductCD": ["W", "W", "C", "W", "C", "W", "R", "R", "R", "H", "R", "H"],
            "card4": ["visa"] * 6 + ["mastercard"] * 6,
        }
    )
    train_identity = pd.DataFrame(
        {
            "TransactionID": range(1, 13),
            "DeviceType": ["desktop"] * 6 + ["mobile"] * 6,
        }
    )
    train_transaction.to_csv(tmp_path / "train_transaction.csv", index=False)
    train_identity.to_csv(tmp_path / "train_identity.csv", index=False)

    result = run_model_training(
        ModelTrainingConfig(
            data_dir=str(tmp_path),
            output_dir=str(tmp_path / "outputs"),
            selected_features_path=None,
            max_features=4,
            epochs=20,
            hidden_units=4,
        )
    )

    metrics = pd.read_csv(tmp_path / "outputs" / "model_comparison_metrics.csv")
    confusion = pd.read_csv(tmp_path / "outputs" / "confusion_matrices.csv")
    submission = pd.read_csv(tmp_path / "outputs" / "validation_submission.csv")
    predictions = pd.read_csv(tmp_path / "outputs" / "validation_predictions.csv")

    assert result["train_rows"] == 8
    assert set(metrics["model"]) == {"rules_based", "neural_network", "random_contender"}
    assert {"tn", "fp", "fn", "tp"}.issubset(confusion.columns)
    assert submission.columns.tolist() == ["TransactionID", "isFraud"]
    assert len(submission) == result["validation_rows"]
    assert predictions[["rules_based", "neural_network", "random_contender"]].shape == (4, 3)


def test_neural_network_prediction_output_shape_and_submission_writer(tmp_path):
    X = pd.DataFrame({"amount": [0.0, 0.1, 0.9, 1.0], "flag": [0, 0, 1, 1]})
    y = pd.Series([0, 0, 1, 1])
    model = NeuralNetworkFraudClassifier(hidden_units=3, epochs=10, random_state=0).fit(X, y)
    scores = model.predict_proba(X)

    assert scores.shape == (4,)
    assert ((scores >= 0.0) & (scores <= 1.0)).all()

    predictions = pd.DataFrame({"TransactionID": [1, 2, 3, 4], "nn": scores})
    write_submission(predictions, tmp_path / "submission.csv")
    submission = pd.read_csv(tmp_path / "submission.csv")

    assert submission.columns.tolist() == ["TransactionID", "isFraud"]
    assert len(compare_models(pd.DataFrame({"TransactionID": [1, 2], "actual": [0, 1], "nn": [0.2, 0.8]}))[0]) == 1
