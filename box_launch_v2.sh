#!/usr/bin/env bash
# box_launch_v2.sh — start the 4 production workers (model-stops spec).
# Servers must already be up on 8937-8940 (verify: bash box_status_v2.sh).
# Requires BLANDV2_GO=1 (deliberate: this is the production start switch).
set -eu
[ "${BLANDV2_GO:-0}" = "1" ] || { echo "refusing: set BLANDV2_GO=1"; exit 1; }
cd /workspace/blandv2
mkdir -p battery_data_v2
for i in 0 1 2 3; do
  port=$((8937 + i))
  curl -s -m 5 "http://localhost:$port/health" | grep -q ok || { echo "server $port unhealthy"; exit 1; }
done
for i in 0 1 2 3; do
  BLAND_URL="http://localhost:$((8937 + i))/v1/completions" \
  BLAND_SHARD="$i/4" BLAND_WORKERS=8 \
  BLAND_CAP=800 BLAND_ATTEMPTS=60 MG_LEN_MAX=700 \
  BLAND_OUT=battery_data_v2/bland_prose_stopped \
  setsid nohup /venv/main/bin/python build_dataset_v2.py \
    > "worker$i.log" 2>&1 < /dev/null &
done
sleep 2
grep -H . worker*.log | head -8
echo "SMOKE_READY workers=4 cap=800 attempts=60 len_max=700 spec=model-stops"
