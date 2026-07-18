# Fraud Model Context Report

## Run Summary

- **rows**: 590540
- **train_rows**: 442905
- **validation_rows**: 147635
- **features_used**: 60
- **test_submission_written**: True

## Model Performance

| model | roc_auc | pr_auc | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- |
| rules_based | 0.504542 | 0.0477038 | 0.959184 | 0.00909795 | 0.0180249 |
| decision_tree | 0.764344 | 0.165728 | 0.0996244 | 0.652149 | 0.172845 |
| random_forest | 0.811294 | 0.292964 | 0.110817 | 0.648471 | 0.189287 |
| neural_network | 0.766541 | 0.158041 | 0.100424 | 0.60453 | 0.172237 |
| random_contender | 0.50003 | 0.0346655 | 0 | 0 | 0 |

## Confusion Matrix Context

- **rules_based**: TP=47, FP=2, FN=5119, TN=142467 at threshold 0.5.
- **decision_tree**: TP=3369, FP=30448, FN=1797, TN=112021 at threshold 0.5.
- **random_forest**: TP=3350, FP=26880, FN=1816, TN=115589 at threshold 0.5.
- **neural_network**: TP=3123, FP=27975, FN=2043, TN=114494 at threshold 0.5.
- **random_contender**: TP=0, FP=0, FN=5166, TN=142469 at threshold 0.5.

## Validation Prediction Context

- Validation rows: 147635
- Fraud positives in validation rows: 5166
- Fraud-positive rate: 0.034992

## Decision Tree Leaf Context

| path | prediction | samples | fraud_rate |
| --- | --- | --- | --- |
| V74 <= 0 AND D3 <= 1 AND M4 == __missing__ | 0.321108 | 17022 | 0.0168605 |
| V74 <= 0 AND D3 <= 1 AND M4 != __missing__ | 0.656157 | 39497 | 0.0647138 |
| V74 <= 0 AND D3 > 1 AND M4 == __missing__ | 0.146673 | 137239 | 0.00619357 |
| V74 <= 0 AND D3 > 1 AND M4 != __missing__ | 0.361023 | 137138 | 0.0200747 |
| V74 > 0 AND C11 <= 1 AND V258 <= 1 | 0.431022 | 67109 | 0.0267326 |
| V74 > 0 AND C11 <= 1 AND V258 > 1 | 0.74448 | 20325 | 0.0955474 |
| V74 > 0 AND C11 > 1 AND C11 <= 2 | 0.779892 | 12438 | 0.113845 |
| V74 > 0 AND C11 > 1 AND C11 > 2 | 0.928843 | 12137 | 0.321249 |

## Decision Tree Decision Impacts

Each row shows how one branch changes the predicted fraud probability relative to the parent node.

| path_before_decision | decision | parent_prediction | child_prediction | prediction_delta | child_samples | child_fraud_rate |
| --- | --- | --- | --- | --- | --- | --- |
| V74 <= 0 AND D3 <= 1 | M4 == __missing__ | 0.593629 | 0.321108 | -0.27252 | 17022 | 0.0168605 |
| V74 <= 0 | D3 <= 1 | 0.353983 | 0.593629 | 0.239646 | 56519 | 0.0503017 |
| root | V74 > 0 | 0.5 | 0.707991 | 0.207991 | 112009 | 0.080806 |
| V74 > 0 AND C11 <= 1 | V258 > 1 | 0.551787 | 0.74448 | 0.192693 | 20325 | 0.0955474 |
| V74 > 0 | C11 > 1 | 0.707991 | 0.88387 | 0.175879 | 24575 | 0.216277 |
| V74 > 0 | C11 <= 1 | 0.707991 | 0.551787 | -0.156204 | 87434 | 0.0427294 |
| root | V74 <= 0 | 0.5 | 0.353983 | -0.146017 | 330896 | 0.0194804 |
| V74 <= 0 AND D3 > 1 | M4 == __missing__ | 0.268465 | 0.146673 | -0.121792 | 137239 | 0.00619357 |
| V74 > 0 AND C11 <= 1 | V258 <= 1 | 0.551787 | 0.431022 | -0.120765 | 67109 | 0.0267326 |
| V74 > 0 AND C11 > 1 | C11 <= 2 | 0.88387 | 0.779892 | -0.103978 | 12438 | 0.113845 |
| V74 <= 0 AND D3 > 1 | M4 != __missing__ | 0.268465 | 0.361023 | 0.0925585 | 137138 | 0.0200747 |
| V74 <= 0 | D3 > 1 | 0.353983 | 0.268465 | -0.0855176 | 274377 | 0.0131316 |
| V74 <= 0 AND D3 <= 1 | M4 != __missing__ | 0.593629 | 0.656157 | 0.0625282 | 39497 | 0.0647138 |
| V74 > 0 AND C11 > 1 | C11 > 2 | 0.88387 | 0.928843 | 0.0449737 | 12137 | 0.321249 |
