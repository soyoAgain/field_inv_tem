#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=================================="
echo " TEM 1D Inversion — Point 11"
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
echo " Done."
echo "=================================="
