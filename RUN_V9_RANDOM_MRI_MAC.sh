#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m src.step26_random_mri_uncertainty "$@"
