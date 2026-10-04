#!/bin/bash
set -e
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
unset LD_LIBRARY_PATH
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
cd /workspace/stoix-navix
exec /workspace/stoix-navix/.venv/bin/python benchmark_navix.py
