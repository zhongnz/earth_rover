#!/usr/bin/env bash
set -euo pipefail

echo "Installing baseline ORB-SLAM3 native dependencies for Ubuntu..."
echo
echo "This script installs the packages the repo can reasonably automate."
echo "If Pangolin is not available from apt on your distro, install it from source."
echo

sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  pkg-config \
  python3 \
  python3-dev \
  python3-numpy \
  libeigen3-dev \
  libopencv-dev \
  libglew-dev

if apt-cache show libpangolin-dev >/dev/null 2>&1; then
  sudo apt install -y libpangolin-dev
else
  cat <<'MSG'

libpangolin-dev is not available from apt on this distro.
Install Pangolin from source before attempting to build ORB-SLAM3 itself:

  scripts/install_pangolin_from_source.sh

Upstream reference:
  https://github.com/stevenlovegrove/Pangolin

MSG
fi

echo
echo "Done. Verify with:"
echo "  scripts/check_orbslam3_toolchain.sh"
