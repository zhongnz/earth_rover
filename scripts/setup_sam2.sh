#!/usr/bin/env bash
set -euo pipefail

SAM2_REPO_COMMIT="2b90b9f5ceec907a1c18123530e92e794ad901a4"
ROOT_DIR=".models/sam2"
VARIANT="sam2.1_hiera_large"
FORCE=0
CONFIG_SHA256=""
CHECKPOINT_SHA256=""

usage() {
  cat <<'EOF'
Download SAM2 config + checkpoint into a local, git-ignored model cache.

Usage:
  scripts/setup_sam2.sh [options]

Options:
  --variant <name>              Model variant (default: sam2.1_hiera_large)
  --root <dir>                  Output root directory (default: .models/sam2)
  --force                       Re-download files even if they already exist
  --config-sha256 <hex>         Optional expected SHA256 for config file
  --checkpoint-sha256 <hex>     Optional expected SHA256 for checkpoint file
  -h, --help                    Show this help

Variants:
  sam2.1_hiera_tiny
  sam2.1_hiera_small
  sam2.1_hiera_base_plus
  sam2.1_hiera_large
  sam2_hiera_tiny
  sam2_hiera_small
  sam2_hiera_base_plus
  sam2_hiera_large
EOF
}

abspath() {
  python - "$1" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
}

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
    return
  fi
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
    return
  fi
  echo "ERROR: neither sha256sum nor shasum found." >&2
  exit 1
}

download_file() {
  local url="$1"
  local dst="$2"

  if [[ -f "$dst" && "$FORCE" -eq 0 ]]; then
    echo "skip (exists): $dst"
    return
  fi

  mkdir -p "$(dirname "$dst")"
  if command -v curl >/dev/null 2>&1; then
    echo "download: $url"
    curl -fL "$url" -o "$dst"
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    echo "download: $url"
    wget -O "$dst" "$url"
    return
  fi

  echo "ERROR: install curl or wget." >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    --root)
      ROOT_DIR="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --config-sha256)
      CONFIG_SHA256="$2"
      shift 2
      ;;
    --checkpoint-sha256)
      CHECKPOINT_SHA256="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

CONFIG_REL=""
CHECKPOINT_URL=""
CHECKPOINT_NAME=""
HYDRA_CFG=""

case "$VARIANT" in
  sam2.1_hiera_tiny)
    CONFIG_REL="sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
    HYDRA_CFG="configs/sam2.1/sam2.1_hiera_t.yaml"
    CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
    CHECKPOINT_NAME="sam2.1_hiera_tiny.pt"
    ;;
  sam2.1_hiera_small)
    CONFIG_REL="sam2/configs/sam2.1/sam2.1_hiera_s.yaml"
    HYDRA_CFG="configs/sam2.1/sam2.1_hiera_s.yaml"
    CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
    CHECKPOINT_NAME="sam2.1_hiera_small.pt"
    ;;
  sam2.1_hiera_base_plus)
    CONFIG_REL="sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
    HYDRA_CFG="configs/sam2.1/sam2.1_hiera_b+.yaml"
    CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
    CHECKPOINT_NAME="sam2.1_hiera_base_plus.pt"
    ;;
  sam2.1_hiera_large)
    CONFIG_REL="sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    HYDRA_CFG="configs/sam2.1/sam2.1_hiera_l.yaml"
    CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    CHECKPOINT_NAME="sam2.1_hiera_large.pt"
    ;;
  sam2_hiera_tiny)
    CONFIG_REL="sam2/configs/sam2/sam2_hiera_t.yaml"
    HYDRA_CFG="configs/sam2/sam2_hiera_t.yaml"
    CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
    CHECKPOINT_NAME="sam2_hiera_tiny.pt"
    ;;
  sam2_hiera_small)
    CONFIG_REL="sam2/configs/sam2/sam2_hiera_s.yaml"
    HYDRA_CFG="configs/sam2/sam2_hiera_s.yaml"
    CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
    CHECKPOINT_NAME="sam2_hiera_small.pt"
    ;;
  sam2_hiera_base_plus)
    CONFIG_REL="sam2/configs/sam2/sam2_hiera_b+.yaml"
    HYDRA_CFG="configs/sam2/sam2_hiera_b+.yaml"
    CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"
    CHECKPOINT_NAME="sam2_hiera_base_plus.pt"
    ;;
  sam2_hiera_large)
    CONFIG_REL="sam2/configs/sam2/sam2_hiera_l.yaml"
    HYDRA_CFG="configs/sam2/sam2_hiera_l.yaml"
    CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    CHECKPOINT_NAME="sam2_hiera_large.pt"
    ;;
  *)
    echo "Unsupported variant: $VARIANT" >&2
    usage
    exit 2
    ;;
esac

CONFIG_URL="https://raw.githubusercontent.com/facebookresearch/sam2/${SAM2_REPO_COMMIT}/${CONFIG_REL}"
CONFIG_BASENAME="$(basename "$CONFIG_REL")"
CONFIG_PATH="${ROOT_DIR}/configs/${CONFIG_BASENAME}"
CHECKPOINT_PATH="${ROOT_DIR}/checkpoints/${CHECKPOINT_NAME}"

download_file "$CONFIG_URL" "$CONFIG_PATH"
download_file "$CHECKPOINT_URL" "$CHECKPOINT_PATH"

if [[ -n "$CONFIG_SHA256" ]]; then
  ACTUAL_CONFIG_SHA256="$(sha256_file "$CONFIG_PATH")"
  if [[ "$ACTUAL_CONFIG_SHA256" != "$CONFIG_SHA256" ]]; then
    echo "ERROR: config sha256 mismatch." >&2
    echo "expected: $CONFIG_SHA256" >&2
    echo "actual:   $ACTUAL_CONFIG_SHA256" >&2
    exit 1
  fi
fi

if [[ -n "$CHECKPOINT_SHA256" ]]; then
  ACTUAL_CHECKPOINT_SHA256="$(sha256_file "$CHECKPOINT_PATH")"
  if [[ "$ACTUAL_CHECKPOINT_SHA256" != "$CHECKPOINT_SHA256" ]]; then
    echo "ERROR: checkpoint sha256 mismatch." >&2
    echo "expected: $CHECKPOINT_SHA256" >&2
    echo "actual:   $ACTUAL_CHECKPOINT_SHA256" >&2
    exit 1
  fi
fi

ABS_CONFIG_PATH="$(abspath "$CONFIG_PATH")"
ABS_CHECKPOINT_PATH="$(abspath "$CHECKPOINT_PATH")"

echo ""
echo "SAM2 assets ready:"
echo "  variant:    $VARIANT"
echo "  config key: $HYDRA_CFG"
echo "  config:     $ABS_CONFIG_PATH"
echo "  checkpoint: $ABS_CHECKPOINT_PATH"
echo ""
echo "Export for this shell:"
echo "  export SAM2_MODEL_CFG=\"$HYDRA_CFG\""
echo "  export SAM2_CHECKPOINT=\"$ABS_CHECKPOINT_PATH\""
echo "  export SAM2_DEVICE=auto"
echo ""
echo "Run preflight:"
echo "  python -m erc_autonomy.check_sam2"
echo ""
echo "Run GPS stack:"
echo "  python -m erc_autonomy.run_gps --traversability-backend sam2"
