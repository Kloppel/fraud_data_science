#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
setsid bash run_full_fraud_workflow_then_sleep.sh > logs/full_fraud_workflow_nohup.log 2>&1 < /dev/null &
echo "$!" > logs/full_fraud_workflow.pid
echo "Started full fraud workflow PID $(cat logs/full_fraud_workflow.pid)"
