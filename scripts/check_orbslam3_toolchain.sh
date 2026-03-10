#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORB_ROOT="${ROOT}/external/ORB_SLAM3"
TMP_DIR=""
PANGOLIN_PREFIX="${PANGOLIN_PREFIX:-${ROOT}/external/pangolin-install}"

missing=0

check_bin() {
  local name="$1"
  if ! command -v "${name}" >/dev/null 2>&1; then
    echo "Missing required tool: ${name}" >&2
    missing=1
  fi
}

check_bin cmake
check_bin c++
check_bin pkg-config

if [[ ! -d "${ORB_ROOT}" ]]; then
  echo "Missing ORB_SLAM3 source tree at ${ORB_ROOT}" >&2
  echo "Run scripts/setup_orbslam3.sh first." >&2
  missing=1
fi

cleanup() {
  if [[ -n "${TMP_DIR}" && -d "${TMP_DIR}" ]]; then
    rm -rf "${TMP_DIR}"
  fi
}
trap cleanup EXIT

if [[ "${missing}" -eq 0 ]]; then
  TMP_DIR="$(mktemp -d)"
  cat > "${TMP_DIR}/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.16)
project(orbslam3_toolchain_probe LANGUAGES CXX)
find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
add_executable(probe main.cpp)
target_link_libraries(probe PRIVATE ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
EOF
  cat > "${TMP_DIR}/main.cpp" <<'EOF'
int main() { return 0; }
EOF

  cmake_env=()
  if [[ -d "${PANGOLIN_PREFIX}" ]]; then
    cmake_env+=(env "CMAKE_PREFIX_PATH=${PANGOLIN_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}")
  fi

  if ! "${cmake_env[@]}" cmake -S "${TMP_DIR}" -B "${TMP_DIR}/build" >/tmp/orbslam3_toolchain_check.log 2>&1; then
    if ! grep -q "Could not find a package configuration file provided by \"OpenCV\"" /tmp/orbslam3_toolchain_check.log; then
      :
    else
      echo "CMake could not find OpenCV; install OpenCV development packages (e.g. libopencv-dev)." >&2
      missing=1
    fi

    if ! grep -q "Could not find a package configuration file provided by \"Pangolin\"" /tmp/orbslam3_toolchain_check.log; then
      :
    else
      echo "CMake could not find Pangolin; install Pangolin dev files or run scripts/install_pangolin_from_source.sh." >&2
      missing=1
    fi

    if [[ "${missing}" -eq 0 ]]; then
      echo "CMake toolchain probe failed for an unexpected reason. See /tmp/orbslam3_toolchain_check.log" >&2
      missing=1
    fi
  fi
fi

if [[ "${missing}" -ne 0 ]]; then
  exit 1
fi

echo "ORB-SLAM3 sidecar toolchain looks present."
