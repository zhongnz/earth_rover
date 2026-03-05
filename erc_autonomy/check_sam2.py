from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SAM2_CFG_MAP = {
    "sam2.1_hiera_t.yaml": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_s.yaml": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_b+.yaml": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_l.yaml": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "sam2_hiera_t.yaml": "configs/sam2/sam2_hiera_t.yaml",
    "sam2_hiera_s.yaml": "configs/sam2/sam2_hiera_s.yaml",
    "sam2_hiera_b+.yaml": "configs/sam2/sam2_hiera_b+.yaml",
    "sam2_hiera_l.yaml": "configs/sam2/sam2_hiera_l.yaml",
}


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except Exception:
        return


def _resolve_device(requested: str) -> tuple[str, str]:
    requested = requested.strip().lower()
    if requested != "auto":
        return requested, "explicit"

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "torch.cuda.is_available()"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", "torch.backends.mps.is_available()"
    except Exception:
        pass
    return "cpu", "fallback"


def _normalize_model_cfg(model_cfg: str) -> str:
    cfg = (model_cfg or "").strip()
    if not cfg:
        return cfg
    if cfg in _SAM2_CFG_MAP.values():
        return cfg
    basename = os.path.basename(cfg)
    return _SAM2_CFG_MAP.get(basename, cfg)


def _model_cfg_status(model_cfg: str) -> tuple[bool, str]:
    cfg = (model_cfg or "").strip()
    if not cfg:
        return False, "empty path"
    path = Path(cfg).expanduser()
    if path.exists():
        if not path.is_file():
            return False, "path is not a file"
        if not os.access(path, os.R_OK):
            return False, "file is not readable"
        return True, "local file"

    normalized = _normalize_model_cfg(cfg)
    if normalized in _SAM2_CFG_MAP.values():
        return True, f"hydra key ({normalized})"
    return False, "file does not exist"


def _exists_readable_file(path_str: str) -> tuple[bool, str]:
    if not path_str:
        return False, "empty path"
    path = Path(path_str).expanduser()
    if not path.exists():
        return False, "file does not exist"
    if not path.is_file():
        return False, "path is not a file"
    if not os.access(path, os.R_OK):
        return False, "file is not readable"
    return True, "local file"


def parse_args() -> argparse.Namespace:
    _load_env()
    p = argparse.ArgumentParser(description="Preflight checks for ERC SAM2 backend.")
    p.add_argument(
        "--sam2-model-cfg",
        default=os.getenv("SAM2_MODEL_CFG", ""),
        help="SAM2 model config path (default: env SAM2_MODEL_CFG)",
    )
    p.add_argument(
        "--sam2-checkpoint",
        default=os.getenv("SAM2_CHECKPOINT", ""),
        help="SAM2 checkpoint path (default: env SAM2_CHECKPOINT)",
    )
    p.add_argument(
        "--sam2-device",
        default=os.getenv("SAM2_DEVICE", "auto"),
        choices=["auto", "cpu", "cuda", "mps"],
        help="Requested SAM2 device (default: env SAM2_DEVICE or auto)",
    )
    p.add_argument(
        "--probe-load",
        action="store_true",
        help="Attempt to import/build SAM2 model with provided config/checkpoint",
    )
    return p.parse_args()


def _try_probe_build(model_cfg: str, checkpoint: str, device: str) -> tuple[bool, str]:
    try:
        try:
            from sam2.build_sam import build_sam2  # type: ignore
        except Exception:
            from sam2.build_sam2 import build_sam2  # type: ignore
    except Exception as exc:
        return False, f"SAM2 import failed: {exc}"

    try:
        _ = build_sam2(_normalize_model_cfg(model_cfg), checkpoint, device=device)
    except Exception as exc:
        return False, f"SAM2 build failed: {exc}"
    return True, "ok"


def main() -> int:
    args = parse_args()

    cfg_ok, cfg_msg = _model_cfg_status(args.sam2_model_cfg)
    ckpt_ok, ckpt_msg = _exists_readable_file(args.sam2_checkpoint)
    resolved_device, reason = _resolve_device(args.sam2_device)

    print("SAM2 preflight")
    print(f"- SAM2_MODEL_CFG: {args.sam2_model_cfg or '(unset)'}")
    print(f"  status: {'OK' if cfg_ok else 'FAIL'} ({cfg_msg})")
    print(f"- SAM2_CHECKPOINT: {args.sam2_checkpoint or '(unset)'}")
    print(f"  status: {'OK' if ckpt_ok else 'FAIL'} ({ckpt_msg})")
    print(f"- SAM2_DEVICE request: {args.sam2_device}")
    print(f"  resolved device: {resolved_device} ({reason})")

    if not (cfg_ok and ckpt_ok):
        print("")
        print("Fix missing assets first, for example:")
        print("  scripts/setup_sam2.sh --variant sam2.1_hiera_large")
        return 1

    if args.probe_load:
        ok, msg = _try_probe_build(
            model_cfg=args.sam2_model_cfg,
            checkpoint=args.sam2_checkpoint,
            device=resolved_device,
        )
        print(f"- SAM2 build probe: {'OK' if ok else 'FAIL'} ({msg})")
        if not ok:
            return 1

    print("")
    print("Ready to run:")
    print("  python -m erc_autonomy.run_gps --traversability-backend sam2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
