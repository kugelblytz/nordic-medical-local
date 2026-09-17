#!/usr/bin/env bash
set -euo pipefail

CUDA_LIBS="$(python3 - <<'PY'
import os
import nvidia.cublas.lib
import nvidia.cudnn.lib

paths = []
for module in (nvidia.cublas.lib, nvidia.cudnn.lib):
    if hasattr(module, '__path__'):
        paths.append(next(iter(module.__path__)))
    else:
        paths.append(os.path.dirname(module.__file__))

print(':'.join(paths))
PY
)"

export LD_LIBRARY_PATH="${CUDA_LIBS}:${LD_LIBRARY_PATH:-}"
exec python3 api.py
