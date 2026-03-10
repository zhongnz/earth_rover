#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/external/orbslam3_sidecar/build/orbslam3_sidecar"

if [[ ! -x "${BIN}" ]]; then
  echo "Missing built sidecar binary at ${BIN}" >&2
  echo "Run scripts/build_orbslam3_sidecar.sh first." >&2
  exit 1
fi

exec "${BIN}" "$@"
