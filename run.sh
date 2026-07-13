#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

point_name="${1:-}"
if [[ -z "$point_name" ]]; then
  echo "用法: ./run.sh 测点i"
  exit 1
fi

export TARGET_POINT="$point_name"
export SOURCE_DATA_PATH="/Users/xiechushu/project/EM_app/TEM_app/data/sy6/$point_name"
results_dir="$(pwd)/results/$point_name"

echo "=================================="
echo " TEM 1D Inversion — $TARGET_POINT"
echo " SOURCE_DATA_PATH=$SOURCE_DATA_PATH"
echo " RESULTS_DIR=$results_dir"
echo "=================================="
echo

echo "[0/3] Saving waveform config ..."
python save_waveform.py

echo
echo "[1/3] Log-spaced time gating ..."
python save_gated_data.py

echo
echo "[2/4] DLS inversion ..."
python inv_dls.py

echo
echo "[3/4] Plotting results ..."
python plot_inv.py

echo
echo "=================================="
echo " Done: $TARGET_POINT"
echo "=================================="
