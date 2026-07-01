#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/czc/satellite_guided_ddcolor}"
CONFIG="${CONFIG:-configs/satellite_color_bottleneck_v14_server.yaml}"
EXP_NAME="${EXP_NAME:-satellite_color_bottleneck_v14_server}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_DIR"
mkdir -p outputs/logs checkpoints

STDOUT_LOG="outputs/logs/${EXP_NAME}.stdout.log"
STDERR_LOG="outputs/logs/${EXP_NAME}.stderr.log"

if pgrep -af "train.py .*${EXP_NAME}" >/dev/null; then
  echo "Training process for ${EXP_NAME} is already running:"
  pgrep -af "train.py .*${EXP_NAME}"
  exit 0
fi

nohup "$PYTHON_BIN" -u train.py \
  --config "$CONFIG" \
  --exp_name "$EXP_NAME" \
  > "$STDOUT_LOG" \
  2> "$STDERR_LOG" &

PID="$!"
echo "Started ${EXP_NAME}"
echo "PID: ${PID}"
echo "STDOUT: ${PROJECT_DIR}/${STDOUT_LOG}"
echo "STDERR: ${PROJECT_DIR}/${STDERR_LOG}"
echo
echo "Monitor:"
echo "  tail -f ${PROJECT_DIR}/${STDOUT_LOG}"
echo "  tail -f ${PROJECT_DIR}/${STDERR_LOG}"
echo "  nvidia-smi"
