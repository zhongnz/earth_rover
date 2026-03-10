#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${ROOT}/external/ORB_SLAM3"
REPO_URL="https://github.com/UZ-SLAMLab/ORB_SLAM3.git"
TARBALL_URL="https://codeload.github.com/UZ-SLAMLab/ORB_SLAM3/tar.gz/refs/heads/master"
TMP_TARBALL="${ROOT}/external/ORB_SLAM3.tar.gz"

is_valid_checkout() {
  [[ -f "${TARGET}/include/System.h" || -f "${TARGET}/System.h" ]]
}

if is_valid_checkout; then
  echo "ORB_SLAM3 source already present at ${TARGET}"
  exit 0
fi

rm -rf "${TARGET}"
mkdir -p "${ROOT}/external"

echo "Fetching ORB_SLAM3 into ${TARGET}"
echo "Primary method: git clone"

if timeout 60 git clone --depth 1 "${REPO_URL}" "${TARGET}"; then
  if is_valid_checkout; then
    echo "ORB_SLAM3 checkout ready via git clone."
    exit 0
  fi
  echo "Git clone completed but expected files were missing; falling back to tarball." >&2
  rm -rf "${TARGET}"
else
  echo "Git clone failed or timed out; falling back to tarball." >&2
  rm -rf "${TARGET}"
fi

echo "Fallback method: GitHub tarball"
curl -fL --progress-bar -o "${TMP_TARBALL}" "${TARBALL_URL}"
tar -xzf "${TMP_TARBALL}" -C "${ROOT}/external"
rm -f "${TMP_TARBALL}"

EXTRACTED_DIR="$(find "${ROOT}/external" -maxdepth 1 -type d -name 'ORB_SLAM3-*' | head -1)"
if [[ -z "${EXTRACTED_DIR}" ]]; then
  echo "Failed to locate extracted ORB_SLAM3 tarball contents." >&2
  exit 1
fi
mv "${EXTRACTED_DIR}" "${TARGET}"

if ! is_valid_checkout; then
  echo "ORB_SLAM3 source fetch completed, but expected headers are missing in ${TARGET}." >&2
  exit 1
fi

echo

echo "ORB_SLAM3 source ready."
echo "Next steps:"
echo "  1. Install ORB-SLAM3 system dependencies on a native Linux machine."
echo "  2. Provide a real camera calibration file."
echo "  3. Build the sidecar with scripts/build_orbslam3_sidecar.sh"
