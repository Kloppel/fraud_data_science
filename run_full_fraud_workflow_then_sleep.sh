#!/usr/bin/env bash
set -euo pipefail
set -x

trap 'echo "Full fraud workflow failed at line ${LINENO} with exit code $?."' ERR

mkdir -p logs outputs/full_exploration outputs/full_feature_analysis outputs/full_model_training

START_TS="$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Started full fraud workflow at ${START_TS}"

PYTHON_CMD=(/home/pbuser/anaconda3/envs/fraud/bin/python)

echo "Running exploration."
"${PYTHON_CMD[@]}" fraud_exploration.py \
  --data-dir data \
  --output-dir outputs/full_exploration \
  --similarity-sample-size 30

echo "Running feature analysis."
"${PYTHON_CMD[@]}" fraud_feature_analysis.py \
  --data-dir data \
  --output-dir outputs/full_feature_analysis

echo "Running model training."
"${PYTHON_CMD[@]}" fraud_model_training.py \
  --data-dir data \
  --output-dir outputs/full_model_training \
  --selected-features-path outputs/full_feature_analysis/selected_features.csv

echo "Running result plots."
"${PYTHON_CMD[@]}" fraud_results_plots.py \
  --exploration-dir outputs/full_exploration \
  --feature-dir outputs/full_feature_analysis \
  --model-dir outputs/full_model_training \
  --output-dir outputs/full_results_plots

echo "Running model context."
"${PYTHON_CMD[@]}" fraud_model_context.py \
  --model-dir outputs/full_model_training \
  --output-dir outputs/full_model_context

END_TS="$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Finished full fraud workflow at ${END_TS}"

echo "Staging result files."
git add \
  outputs/full_exploration \
  outputs/full_feature_analysis \
  outputs/full_model_training \
  outputs/full_results_plots \
  outputs/full_model_context \
  logs/

if git diff --cached --quiet; then
  echo "No result-file changes to commit."
else
  echo "Committing result files."
  git commit -m "$(cat <<EOF
Refresh full fraud workflow outputs

Full run started ${START_TS}, finished ${END_TS}.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
  echo "Pushing result files."
  git push origin main
fi

echo "Waiting 120 seconds before suspend."
sleep 120
systemctl suspend
