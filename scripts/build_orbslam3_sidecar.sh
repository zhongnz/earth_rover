#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIDECAR_DIR="${ROOT}/external/orbslam3_sidecar"
BUILD_DIR="${SIDECAR_DIR}/build"
ORB_ROOT="${ROOT}/external/ORB_SLAM3"
PANGOLIN_PREFIX="${PANGOLIN_PREFIX:-${ROOT}/external/pangolin-install}"

"${ROOT}/scripts/check_orbslam3_toolchain.sh"

mkdir -p "${BUILD_DIR}"
if [[ -d "${PANGOLIN_PREFIX}" ]]; then
  export CMAKE_PREFIX_PATH="${PANGOLIN_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
fi

cmake -S "${SIDECAR_DIR}" -B "${BUILD_DIR}" -DORB_SLAM3_ROOT="${ORB_ROOT}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"
