#!/usr/bin/env bash
# Live momentum portfolio at http://localhost:8502 (auto-refreshes data on the 1st of each month)
cd "$(dirname "$0")" && exec .venv/bin/streamlit run portfolio_app.py --server.port 8502 "$@"
