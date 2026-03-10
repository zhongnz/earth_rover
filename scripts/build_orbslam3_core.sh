#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORB_ROOT="${ROOT}/external/ORB_SLAM3"
PANGOLIN_PREFIX="${PANGOLIN_PREFIX:-${ROOT}/external/pangolin-install}"
CONDA_PREFIX_LOCAL="${CONDA_PREFIX:-}"

if [[ ! -f "${ORB_ROOT}/build.sh" ]]; then
  echo "ORB-SLAM3 source not found at ${ORB_ROOT}. Run scripts/setup_orbslam3.sh first." >&2
  exit 1
fi

if [[ -d "${PANGOLIN_PREFIX}" ]]; then
  export CMAKE_PREFIX_PATH="${PANGOLIN_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
fi

if [[ -n "${CONDA_PREFIX_LOCAL}" && -d "${CONDA_PREFIX_LOCAL}/include" ]]; then
  export CPLUS_INCLUDE_PATH="${CONDA_PREFIX_LOCAL}/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
fi

if [[ -n "${CONDA_PREFIX_LOCAL}" && -d "${CONDA_PREFIX_LOCAL}/lib" ]]; then
  export LIBRARY_PATH="${CONDA_PREFIX_LOCAL}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX_LOCAL}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [[ -n "${CONDA_PREFIX_LOCAL}" ]]; then
  export OPENSSL_ROOT_DIR="${CONDA_PREFIX_LOCAL}"
  export OPENSSL_INCLUDE_DIR="${CONDA_PREFIX_LOCAL}/include"
  export OPENSSL_CRYPTO_LIBRARY="${CONDA_PREFIX_LOCAL}/lib/libcrypto.so"
fi

cd "${ORB_ROOT}"
bash ./build.sh

if [[ ! -f "${ORB_ROOT}/lib/libORB_SLAM3.so" ]]; then
  echo "ORB-SLAM3 build completed without producing lib/libORB_SLAM3.so" >&2
  exit 1
fi

echo "ORB-SLAM3 core library ready:"
echo "  ${ORB_ROOT}/lib/libORB_SLAM3.so"
