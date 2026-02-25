#!/bin/bash
cd "$(dirname "$0")/../.."
uv run python -m train.run --config experiments/v56_triviaqa_recon/config.py
