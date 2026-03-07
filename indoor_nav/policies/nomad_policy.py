"""
NoMaD (Goal-Conditioned Diffusion Policy) integration — SOTA 2025.

Supports two checkpoint formats:
  - TorchScript `.pt` files that directly map (context, goal) -> waypoints
  - Official visualnav-transformer `.pth` checkpoints, provided the upstream
    repository checkout and dependencies are available locally

If neither path can be loaded, the policy degrades to a simple heuristic.
"""

from __future__ import annotations

import logging
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from indoor_nav.configs.config import PolicyConfig
from indoor_nav.policies.base_policy import BasePolicy, PolicyInput, PolicyOutput

logger = logging.getLogger(__name__)

_OFFICIAL_ACTION_STATS = {
    "min": np.array([-2.5, -4.0], dtype=np.float32),
    "max": np.array([5.0, 4.0], dtype=np.float32),
}


class NoMaDPolicy(BasePolicy):
    """
    NoMaD goal-conditioned diffusion policy.

    Predicts waypoints in the robot's egocentric frame, which are then
    converted to (linear, angular) velocities via a simple pursuit controller.
    """

    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg
        self._model = None
        self._context_queue: deque = deque(maxlen=cfg.context_length + 1)
        self._device = None
        self._model_format: Optional[str] = None
        self._official_cfg: dict[str, Any] = {}
        self._official_noise_scheduler = None
        self._official_action_stats = {
            "min": _OFFICIAL_ACTION_STATS["min"].copy(),
            "max": _OFFICIAL_ACTION_STATS["max"].copy(),
        }
        self._official_repo_root: Optional[Path] = None

    def setup(self):
        """Load a TorchScript or official NoMaD checkpoint."""
        try:
            import torch

            self._device = torch.device(
                self.cfg.device if torch.cuda.is_available() else "cpu"
            )
        except ImportError:
            logger.warning("PyTorch not installed. Using heuristic fallback.")
            self._model = None
            self._model_format = None
            return

        model_path = Path(self.cfg.model_path).expanduser()
        backend = getattr(self.cfg, "backend", "nomad").lower()
        logger.info("Loading NoMaD model from %s on %s", model_path, self._device)

        if not model_path.exists():
            logger.warning(
                "NoMaD model not found at %s. "
                "Using heuristic fallback. Download weights from: "
                "https://github.com/robodhruv/visualnav-transformer",
                model_path,
            )
            self._model = None
            self._model_format = None
            return

        if model_path.suffix.lower() == ".pth":
            if backend == "nomad":
                if self._load_official_nomad_checkpoint(torch, model_path):
                    return
            elif backend in {"vint", "gnm"}:
                if self._load_official_supervised_checkpoint(torch, model_path, backend):
                    return
            else:
                logger.warning("Unsupported GNM-family backend for .pth checkpoint: %s", backend)
                return
            self._model = None
            self._model_format = None
            return

        try:
            self._model = torch.jit.load(str(model_path), map_location=self._device)
            self._model.eval()
            self._model_format = "torchscript"
            logger.info("TorchScript NoMaD model loaded successfully.")
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
            logger.warning(
                "NoMaD model at %s could not be loaded as TorchScript (%s). "
                "Using heuristic fallback.",
                model_path,
                exc,
            )
            self._model = None
            self._model_format = None

    def predict(self, obs: PolicyInput) -> PolicyOutput:
        """Predict next action from current observation + goal."""
        self._context_queue.append(obs.front_image.copy())

        if self._model_format == "official":
            return self._predict_official_nomad(obs)
        if self._model_format in {"official_vint", "official_gnm"}:
            return self._predict_official_supervised(obs)
        if self._model is not None:
            return self._predict_torchscript_nomad(obs)
        return self._predict_heuristic(obs)

    def _predict_torchscript_nomad(self, obs: PolicyInput) -> PolicyOutput:
        """Run the legacy TorchScript NoMaD wrapper."""
        import torch

        context = list(self._context_queue)[-self.cfg.context_length :]
        while len(context) < self.cfg.context_length:
            context.insert(0, context[0] if context else obs.front_image.copy())

        size = self.cfg.image_size
        context_tensors = []
        for img in context:
            resized = cv2.resize(img, size)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            context_tensors.append(t)

        goal_resized = cv2.resize(obs.goal_image, self.cfg.goal_image_size)
        goal_rgb = cv2.cvtColor(goal_resized, cv2.COLOR_BGR2RGB)
        goal_tensor = torch.from_numpy(goal_rgb).permute(2, 0, 1).float() / 255.0

        context_batch = torch.stack(context_tensors).unsqueeze(0).to(self._device)
        goal_batch = goal_tensor.unsqueeze(0).to(self._device)

        with torch.no_grad():
            waypoints = self._model(context_batch, goal_batch)

        waypoints = waypoints.cpu().numpy()[0]
        linear, angular = self._waypoints_to_velocity(waypoints)
        linear *= obs.obstacle_speed_factor
        angular += obs.obstacle_steer_bias

        return PolicyOutput(
            linear=float(np.clip(linear, -1, 1)),
            angular=float(np.clip(angular, -1, 1)),
            confidence=0.8,
            waypoints=waypoints,
        )

    def _load_official_supervised_checkpoint(self, torch: Any, model_path: Path, model_name: str) -> bool:
        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not installed. Cannot load official %s checkpoint.", model_name)
            return False

        repo_root = self._resolve_nomad_repo_root(model_path)
        if repo_root is None:
            logger.warning(
                "Could not locate visualnav-transformer checkout for %s. "
                "Set --nomad-repo-root or cfg.policy.nomad_repo_root.",
                model_path,
            )
            return False

        train_root = repo_root / "train"
        train_root_str = str(train_root)
        if train_root.exists() and train_root_str not in sys.path:
            sys.path.insert(0, train_root_str)

        config_path = self._resolve_model_config_path(repo_root, model_name)
        if config_path is None:
            logger.warning(
                "Could not locate official %s config under %s. Set --nomad-config-path if needed.",
                model_name,
                repo_root,
            )
            return False

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                model_cfg = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("Failed to read %s config %s: %s", model_name, config_path, exc)
            return False

        try:
            model = self._build_official_supervised_model(model_cfg, model_name)
        except Exception as exc:
            logger.warning("Failed to construct official %s model: %s", model_name, exc)
            return False

        try:
            checkpoint = self._torch_load_checkpoint(torch, model_path)
            state_dict = self._extract_state_dict_for_supervised_checkpoint(checkpoint)
            model.load_state_dict(state_dict, strict=False)
        except Exception as exc:
            logger.warning("Failed to load official %s checkpoint %s: %s", model_name, model_path, exc)
            return False

        model = model.to(self._device)
        model.eval()

        self._model = model
        self._model_format = f"official_{model_name}"
        self._official_cfg = model_cfg
        self._official_repo_root = repo_root
        self._official_action_stats = self._load_official_action_stats(repo_root)
        logger.info(
            "Official %s checkpoint loaded from %s using config %s",
            model_name,
            model_path,
            config_path,
        )
        return True

    def _load_official_nomad_checkpoint(self, torch: Any, model_path: Path) -> bool:
        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not installed. Cannot load official NoMaD checkpoint.")
            return False

        try:
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        except ImportError:
            logger.warning("diffusers not installed. Cannot load official NoMaD checkpoint.")
            return False

        repo_root = self._resolve_nomad_repo_root(model_path)
        if repo_root is None:
            logger.warning(
                "Could not locate visualnav-transformer checkout for %s. "
                "Set --nomad-repo-root or cfg.policy.nomad_repo_root.",
                model_path,
            )
            return False

        config_path = self._resolve_nomad_config_path(repo_root)
        if config_path is None:
            logger.warning(
                "Could not locate official nomad.yaml under %s. "
                "Set --nomad-config-path if needed.",
                repo_root,
            )
            return False

        train_root = repo_root / "train"
        train_root_str = str(train_root)
        if train_root.exists() and train_root_str not in sys.path:
            sys.path.insert(0, train_root_str)

        diffusion_root = self._resolve_diffusion_policy_root(repo_root)
        if diffusion_root is not None:
            diffusion_root_str = str(diffusion_root)
            if diffusion_root_str not in sys.path:
                sys.path.insert(0, diffusion_root_str)

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                model_cfg = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("Failed to read NoMaD config %s: %s", config_path, exc)
            return False

        try:
            model = self._build_official_nomad_model(model_cfg)
        except Exception as exc:
            logger.warning("Failed to construct official NoMaD model: %s", exc)
            return False

        try:
            checkpoint = self._torch_load_checkpoint(torch, model_path)
            state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            if hasattr(state_dict, "module"):
                state_dict = state_dict.module.state_dict()
            elif hasattr(state_dict, "state_dict"):
                state_dict = state_dict.state_dict()
            if not isinstance(state_dict, dict):
                raise TypeError(f"unsupported checkpoint payload type: {type(state_dict)}")
            model.load_state_dict(state_dict, strict=False)
        except Exception as exc:
            logger.warning("Failed to load official NoMaD checkpoint %s: %s", model_path, exc)
            return False

        model = model.to(self._device)
        model.eval()

        self._model = model
        self._model_format = "official"
        self._official_cfg = model_cfg
        self._official_repo_root = repo_root
        self._official_noise_scheduler = DDPMScheduler(
            num_train_timesteps=int(model_cfg["num_diffusion_iters"]),
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self._official_action_stats = self._load_official_action_stats(repo_root)
        logger.info(
            "Official NoMaD checkpoint loaded from %s using config %s",
            model_path,
            config_path,
        )
        return True

    def _build_official_nomad_model(self, model_cfg: dict[str, Any]):
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        from vint_train.models.nomad.nomad import DenseNetwork, NoMaD

        vision_encoder_name = model_cfg.get("vision_encoder", "nomad_vint")
        if vision_encoder_name == "nomad_vint":
            from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=model_cfg["encoding_size"],
                context_size=model_cfg["context_size"],
                mha_num_attention_heads=model_cfg["mha_num_attention_heads"],
                mha_num_attention_layers=model_cfg["mha_num_attention_layers"],
                mha_ff_dim_factor=model_cfg["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif vision_encoder_name == "vit":
            from vint_train.models.nomad.nomad_vint import replace_bn_with_gn
            from vint_train.models.vint.vit import ViT

            vision_encoder = ViT(
                obs_encoding_size=model_cfg["encoding_size"],
                context_size=model_cfg["context_size"],
                image_size=model_cfg["image_size"],
                patch_size=model_cfg["patch_size"],
                mha_num_attention_heads=model_cfg["mha_num_attention_heads"],
                mha_num_attention_layers=model_cfg["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else:
            raise ValueError(f"unsupported official NoMaD vision_encoder: {vision_encoder_name}")

        noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=model_cfg["encoding_size"],
            down_dims=model_cfg["down_dims"],
            cond_predict_scale=model_cfg["cond_predict_scale"],
        )
        dist_pred_network = DenseNetwork(embedding_dim=model_cfg["encoding_size"])
        return NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )

    def _build_official_supervised_model(self, model_cfg: dict[str, Any], model_name: str):
        if model_name == "vint":
            from vint_train.models.vint.vint import ViNT

            return ViNT(
                context_size=model_cfg["context_size"],
                len_traj_pred=model_cfg["len_traj_pred"],
                learn_angle=model_cfg["learn_angle"],
                obs_encoder=model_cfg["obs_encoder"],
                obs_encoding_size=model_cfg["obs_encoding_size"],
                late_fusion=model_cfg["late_fusion"],
                mha_num_attention_heads=model_cfg["mha_num_attention_heads"],
                mha_num_attention_layers=model_cfg["mha_num_attention_layers"],
                mha_ff_dim_factor=model_cfg["mha_ff_dim_factor"],
            )

        if model_name == "gnm":
            from vint_train.models.gnm.gnm import GNM

            return GNM(
                context_size=model_cfg["context_size"],
                len_traj_pred=model_cfg["len_traj_pred"],
                learn_angle=model_cfg["learn_angle"],
                obs_encoding_size=model_cfg["obs_encoding_size"],
                goal_encoding_size=model_cfg["goal_encoding_size"],
            )

        raise ValueError(f"unsupported supervised model: {model_name}")

    def _extract_state_dict_for_supervised_checkpoint(self, checkpoint: Any) -> dict[str, Any]:
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            loaded_model = checkpoint["model"]
            if hasattr(loaded_model, "module"):
                return loaded_model.module.state_dict()
            if hasattr(loaded_model, "state_dict"):
                return loaded_model.state_dict()
        if isinstance(checkpoint, dict):
            return checkpoint
        if hasattr(checkpoint, "state_dict"):
            return checkpoint.state_dict()
        raise TypeError(f"unsupported checkpoint payload type: {type(checkpoint)}")

    def _torch_load_checkpoint(self, torch: Any, model_path: Path):
        try:
            return torch.load(str(model_path), map_location=self._device, weights_only=False)
        except TypeError:
            return torch.load(str(model_path), map_location=self._device)

    def _resolve_nomad_repo_root(self, model_path: Path) -> Optional[Path]:
        candidates = []

        if self.cfg.nomad_repo_root:
            candidates.append(Path(self.cfg.nomad_repo_root).expanduser())

        env_root = os.getenv("VISUALNAV_TRANSFORMER_ROOT", "")
        if env_root:
            candidates.append(Path(env_root).expanduser())

        for parent in [model_path.parent, *model_path.parents]:
            candidates.append(parent)

        project_root = Path(__file__).resolve().parents[2]
        candidates.extend(
            [
                project_root / "visualnav-transformer",
                project_root / "external" / "visualnav-transformer",
                project_root / "third_party" / "visualnav-transformer",
            ]
        )

        seen = set()
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "train" / "config" / "nomad.yaml").exists():
                return candidate
        return None

    def _resolve_diffusion_policy_root(self, repo_root: Path) -> Optional[Path]:
        candidates = []

        env_root = os.getenv("DIFFUSION_POLICY_ROOT", "")
        if env_root:
            candidates.append(Path(env_root).expanduser())

        for candidate in [
            repo_root.parent / "diffusion_policy",
            Path(__file__).resolve().parents[2] / "external" / "diffusion_policy",
            Path(__file__).resolve().parents[2] / "third_party" / "diffusion_policy",
        ]:
            candidates.append(candidate)

        seen = set()
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "diffusion_policy" / "model" / "diffusion" / "conditional_unet1d.py").exists():
                return candidate
        return None

    def _resolve_model_config_path(self, repo_root: Path, model_name: str) -> Optional[Path]:
        candidates = []

        if self.cfg.nomad_config_path:
            candidates.append(Path(self.cfg.nomad_config_path).expanduser())

        env_path = os.getenv("NOMAD_CONFIG_PATH", "")
        if env_path:
            candidates.append(Path(env_path).expanduser())

        candidates.append(repo_root / "train" / "config" / f"{model_name}.yaml")

        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    def _resolve_nomad_config_path(self, repo_root: Path) -> Optional[Path]:
        return self._resolve_model_config_path(repo_root, "nomad")

    def _load_official_action_stats(self, repo_root: Path) -> dict[str, np.ndarray]:
        try:
            import yaml
        except ImportError:
            return {
                "min": _OFFICIAL_ACTION_STATS["min"].copy(),
                "max": _OFFICIAL_ACTION_STATS["max"].copy(),
            }

        data_cfg_path = repo_root / "train" / "vint_train" / "data" / "data_config.yaml"
        if not data_cfg_path.exists():
            return {
                "min": _OFFICIAL_ACTION_STATS["min"].copy(),
                "max": _OFFICIAL_ACTION_STATS["max"].copy(),
            }

        try:
            with open(data_cfg_path, "r", encoding="utf-8") as f:
                data_cfg = yaml.safe_load(f) or {}
            stats = data_cfg.get("action_stats", {})
            if "min" in stats and "max" in stats:
                return {
                    "min": np.asarray(stats["min"], dtype=np.float32),
                    "max": np.asarray(stats["max"], dtype=np.float32),
                }
        except Exception as exc:
            logger.debug("Failed to load official NoMaD action stats: %s", exc)

        return {
            "min": _OFFICIAL_ACTION_STATS["min"].copy(),
            "max": _OFFICIAL_ACTION_STATS["max"].copy(),
        }

    def _predict_official_nomad(self, obs: PolicyInput) -> PolicyOutput:
        import torch
        from PIL import Image
        from torchvision import transforms

        if self._official_noise_scheduler is None or not self._official_cfg:
            return self._predict_heuristic(obs)

        context_len = int(self._official_cfg.get("context_size", 3)) + 1
        image_size = tuple(int(v) for v in self._official_cfg.get("image_size", (96, 96)))
        pred_horizon = int(self._official_cfg.get("len_traj_pred", self.cfg.action_horizon))
        num_diffusion_iters = int(self._official_cfg.get("num_diffusion_iters", 10))
        num_samples = max(1, int(getattr(self.cfg, "nomad_num_samples", 1)))

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        context = list(self._context_queue)[-context_len:]
        if not context:
            context = [obs.front_image.copy()]
        while len(context) < context_len:
            context.insert(0, context[0])

        obs_images = self._stack_images_for_official(
            context,
            image_size=image_size,
            transform=transform,
            image_cls=Image,
            torch=torch,
        ).to(self._device)
        goal_image = self._stack_images_for_official(
            [obs.goal_image],
            image_size=image_size,
            transform=transform,
            image_cls=Image,
            torch=torch,
        ).to(self._device)

        goal_mask = torch.zeros((1,), dtype=torch.long, device=self._device)

        with torch.no_grad():
            obs_cond = self._model(
                "vision_encoder",
                obs_img=obs_images,
                goal_img=goal_image,
                input_goal_mask=goal_mask,
            )
            obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)
            diffusion_output = torch.randn((num_samples, pred_horizon, 2), device=self._device)

            self._official_noise_scheduler.set_timesteps(num_diffusion_iters)
            for timestep in self._official_noise_scheduler.timesteps:
                if torch.is_tensor(timestep):
                    step_for_model = timestep.reshape(1).repeat(diffusion_output.shape[0]).to(self._device)
                else:
                    step_for_model = torch.full(
                        (diffusion_output.shape[0],),
                        int(timestep),
                        device=self._device,
                        dtype=torch.long,
                    )
                noise_pred = self._model(
                    "noise_pred_net",
                    sample=diffusion_output,
                    timestep=step_for_model,
                    global_cond=obs_cond,
                )
                diffusion_output = self._official_noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=diffusion_output,
                ).prev_sample

        waypoints = self._official_diffusion_to_waypoints(diffusion_output.cpu().numpy())[0]
        linear, angular = self._waypoints_to_velocity(waypoints)
        linear *= obs.obstacle_speed_factor
        angular += obs.obstacle_steer_bias

        return PolicyOutput(
            linear=float(np.clip(linear, -1, 1)),
            angular=float(np.clip(angular, -1, 1)),
            confidence=0.75,
            waypoints=waypoints,
        )

    def _predict_official_supervised(self, obs: PolicyInput) -> PolicyOutput:
        import torch
        from PIL import Image
        from torchvision import transforms

        if not self._official_cfg:
            return self._predict_heuristic(obs)

        context_len = int(self._official_cfg.get("context_size", self.cfg.context_length)) + 1
        image_size = tuple(int(v) for v in self._official_cfg.get("image_size", (85, 64)))

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        context = list(self._context_queue)[-context_len:]
        if not context:
            context = [obs.front_image.copy()]
        while len(context) < context_len:
            context.insert(0, context[0])

        obs_images = self._stack_images_for_official(
            context,
            image_size=image_size,
            transform=transform,
            image_cls=Image,
            torch=torch,
        ).to(self._device)
        goal_image = self._stack_images_for_official(
            [obs.goal_image],
            image_size=image_size,
            transform=transform,
            image_cls=Image,
            torch=torch,
        ).to(self._device)

        with torch.no_grad():
            dist_pred, action_pred = self._model(obs_images, goal_image)

        waypoints = action_pred[0, :, :2].detach().cpu().numpy()
        linear, angular = self._waypoints_to_velocity(waypoints)
        linear *= obs.obstacle_speed_factor
        angular += obs.obstacle_steer_bias

        dist_value = float(dist_pred.reshape(-1)[0].detach().cpu().item()) if dist_pred is not None else 0.0
        confidence = float(np.clip(1.0 / (1.0 + max(0.0, dist_value)), 0.2, 0.8))

        return PolicyOutput(
            linear=float(np.clip(linear, -1, 1)),
            angular=float(np.clip(angular, -1, 1)),
            confidence=confidence,
            waypoints=waypoints,
        )

    def _stack_images_for_official(self, images, image_size, transform, image_cls, torch):
        tensors = []
        for image in images:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = image_cls.fromarray(rgb)
            pil_img = pil_img.resize(image_size)
            tensors.append(transform(pil_img).unsqueeze(0))
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=1)

    def _official_diffusion_to_waypoints(self, diffusion_output: np.ndarray) -> np.ndarray:
        deltas = diffusion_output.reshape(diffusion_output.shape[0], -1, 2)
        stats = self._official_action_stats
        scaled = (deltas + 1.0) / 2.0
        deltas = scaled * (stats["max"] - stats["min"]) + stats["min"]
        return np.cumsum(deltas, axis=1)

    def _predict_heuristic(self, obs: PolicyInput) -> PolicyOutput:
        """
        Heuristic fallback when no trained model is available.

        Strategy: Use goal similarity + trend to decide basic motions.
        - If similarity is increasing -> go forward
        - If similarity is decreasing -> turn to search
        - If high similarity -> slow down and center
        """
        sim = obs.goal_similarity
        trend = obs.goal_trend

        if sim > 0.7:
            linear = 0.2
            angular = 0.0
            confidence = 0.8
        elif trend > 0.01:
            linear = 0.4
            angular = 0.0
            confidence = 0.5
        elif trend < -0.01:
            linear = 0.1
            angular = 0.3
            confidence = 0.3
        else:
            linear = 0.3
            angular = 0.0
            confidence = 0.4

        linear *= obs.obstacle_speed_factor
        angular += obs.obstacle_steer_bias

        return PolicyOutput(
            linear=float(np.clip(linear, -1, 1)),
            angular=float(np.clip(angular, -1, 1)),
            confidence=confidence,
        )

    def _waypoints_to_velocity(self, waypoints: np.ndarray) -> tuple:
        """
        Pure-pursuit controller: follow the first predicted waypoint.

        Waypoints are in egocentric frame: +x = forward, +y = left.
        """
        if waypoints.shape[0] == 0:
            return 0.0, 0.0

        idx = min(1, waypoints.shape[0] - 1)
        target_x = waypoints[idx, 0]
        target_y = waypoints[idx, 1]

        angular = np.clip(target_y * 2.0, -1.0, 1.0)
        linear = np.clip(target_x * 1.5, -1.0, 1.0)
        linear *= max(0.3, 1.0 - abs(angular) * 0.5)
        return float(linear), float(angular)

    def reset(self):
        self._context_queue.clear()
