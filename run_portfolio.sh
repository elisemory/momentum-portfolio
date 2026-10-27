#!/usr/bin/env bash
# Live momentum portfolio at http://localhost:8502 (refreshes prices throughout the US session)
cd "$(dirname "$0")" && exec .venv/bin/streamlit run portfolio_app.py --server.port 8502 "$@"
