#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PANGOLIN_ROOT="${ROOT}/external/Pangolin"
PANGOLIN_REPO="${PANGOLIN_REPO:-https://github.com/stevenlovegrove/Pangolin.git}"
PANGOLIN_REF="${PANGOLIN_REF:-master}"
PREFIX="${PREFIX:-${ROOT}/external/pangolin-install}"
SKIP_APT="${SKIP_APT:-0}"
HAS_SUDO=0
APT_DEPS=(
  build-essential
  cmake
  git
  pkg-config
  libeigen3-dev
  libgl1-mesa-dev
  libglu1-mesa-dev
  libglew-dev
  libepoxy-dev
  libx11-dev
  libjpeg-dev
  libpng-dev
  libavcodec-dev
  libavformat-dev
  libavutil-dev
  libswscale-dev
  libdc1394-dev
  libopenni2-dev
  libboost-dev
  python3-dev
)

if sudo -n true >/dev/null 2>&1; then
  HAS_SUDO=1
fi

if [[ "${SKIP_APT}" != "1" ]]; then
  if [[ "${HAS_SUDO}" -eq 1 ]]; then
    echo "Installing Pangolin build dependencies with apt..."
    sudo apt update
    sudo apt install -y "${APT_DEPS[@]}"
  else
    cat <<'MSG'
No passwordless sudo available; skipping apt dependency installation.
If the Pangolin build fails later, install the missing dev packages manually or
rerun this script in a shell where sudo can prompt.
MSG
  fi
fi

if [[ "${SKIP_APT}" == "1" || "${HAS_SUDO}" -eq 0 ]]; then
  missing_pkgs=()
  for pkg in "${APT_DEPS[@]}"; do
    if ! dpkg-query -W -f='${db:Status-Abbrev}\n' "${pkg}" 2>/dev/null | grep -q '^ii'; then
      missing_pkgs+=("${pkg}")
    fi
  done
  if [[ "${#missing_pkgs[@]}" -ne 0 ]]; then
    echo "Missing Pangolin build dependencies:" >&2
    printf '  %s\n' "${missing_pkgs[@]}" >&2
    echo >&2
    echo "Install them first, for example:" >&2
    echo "  sudo apt install -y ${missing_pkgs[*]}" >&2
    exit 1
  fi
fi

mkdir -p "${ROOT}/external"
if [[ ! -d "${PANGOLIN_ROOT}/.git" ]]; then
  echo "Cloning Pangolin into ${PANGOLIN_ROOT}..."
  git clone --depth 1 --branch "${PANGOLIN_REF}" "${PANGOLIN_REPO}" "${PANGOLIN_ROOT}"
else
  echo "Updating Pangolin checkout in ${PANGOLIN_ROOT}..."
  git -C "${PANGOLIN_ROOT}" fetch --depth 1 origin "${PANGOLIN_REF}"
  git -C "${PANGOLIN_ROOT}" checkout -q FETCH_HEAD
fi

echo "Configuring Pangolin..."
cmake -S "${PANGOLIN_ROOT}" -B "${PANGOLIN_ROOT}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${PREFIX}"

echo "Building Pangolin..."
cmake --build "${PANGOLIN_ROOT}/build" -j"$(nproc)"

echo "Installing Pangolin to ${PREFIX}..."
cmake --install "${PANGOLIN_ROOT}/build"
if [[ "${HAS_SUDO}" -eq 1 && "${PREFIX}" == /usr/* ]]; then
  sudo ldconfig
fi

echo
echo "Installed Pangolin. Verify with:"
echo "  CMAKE_PREFIX_PATH=${PREFIX} scripts/check_orbslam3_toolchain.sh"
