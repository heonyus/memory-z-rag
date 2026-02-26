#!/bin/bash
cd "$(dirname "$0")/../.."
uv run python -m train.run --config experiments/v55_summary_recon/config.py
