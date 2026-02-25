#!/bin/bash
cd "$(dirname "$0")/../.."
uv run python -m train.run --config experiments/v57_summary_50docs/config.py
