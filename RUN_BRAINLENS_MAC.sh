#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! python3 -c 'import streamlit' >/dev/null 2>&1; then
    echo "Installing the lightweight Streamlit web interface..."
    python3 -m pip install streamlit || {
        echo "Unable to install Streamlit in this Python environment."
        echo "Activate your V9 Python environment, then run: python3 -m pip install streamlit"
        exit 1
    }
fi
# Apply the calm light theme only for this launched website.
export STREAMLIT_THEME_BASE=light
export STREAMLIT_THEME_PRIMARY_COLOR="#8AC7EA"
export STREAMLIT_THEME_BACKGROUND_COLOR="#F8FCFF"
export STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR="#EDF7FE"
export STREAMLIT_THEME_TEXT_COLOR="#243A50"
python3 -m streamlit run brainlens_v9.py --server.headless=false
