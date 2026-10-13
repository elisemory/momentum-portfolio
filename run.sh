#!/usr/bin/env bash
# Launch the dashboard at http://localhost:8501
cd "$(dirname "$0")" && exec .venv/bin/streamlit run app.py "$@"
