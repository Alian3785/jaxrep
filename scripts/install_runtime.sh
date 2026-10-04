#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -c runtime-constraints.txt \
  -r craftax-runtime.txt pillow matplotlib pytest
uv pip install --python .venv/bin/python --no-deps -e . -e ./stoa-src
if [[ -f results/craftax-5m/model.msgpack.gz && ! -e results/craftax-5m/model.msgpack ]]; then
  gzip -dk results/craftax-5m/model.msgpack.gz
fi
