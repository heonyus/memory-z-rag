#!/bin/bash
cd "$(dirname "$0")/../.."
uv run python -m train.run --config experiments/v58_text_50docs/config.py
