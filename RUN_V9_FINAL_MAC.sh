#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f results/07_training/best_v9.pt ]; then
  echo 'ERROR: Run from original V9 project folder. Missing results/07_training/best_v9.pt'; exit 1
fi
if [ ! -f results/23_reliability_warning/warning_model.json ]; then
  echo 'ERROR: Install results/23_reliability_warning from V9_Final_Research_System.zip'; exit 1
fi
if ! python3 -c 'import streamlit' >/dev/null 2>&1; then
  echo 'Installing Streamlit (only missing dashboard dependency)...'
  python3 -m pip install 'streamlit>=1.38' || { echo 'Please activate your original V9 Python environment, install streamlit, then retry.'; exit 1; }
fi
python3 -m streamlit run v9_final_reliability_app.py --server.address 127.0.0.1
