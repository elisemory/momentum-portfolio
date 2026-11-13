#!/usr/bin/env bash
# CAN SLIM backtest and daily screen at http://localhost:8503
cd "$(dirname "$0")" && exec .venv/bin/streamlit run canslim_app.py --server.port 8503 "$@"
