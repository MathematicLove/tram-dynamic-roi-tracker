#!/usr/bin/env bash
set -e

python scripts/download_models.py

exec "$@"
