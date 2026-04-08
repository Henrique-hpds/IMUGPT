"""Prompt-motion backend abstractions and the legacy T2M-GPT adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Protocol

import numpy as np


class PromptMotionBackend(Protocol):
    def generate(
        self,
        *,
        prompt_text: str,
        seed: int,
        fps: float,
        duration_hint_sec: float | None = None,
        output_dir: str | Path | None = None,
    ) -> Dict[str, Any]:
        ...


@dataclass(frozen=True)
class LegacyT2MGPTBackendConfig:
    vq_checkpoint_path: str | Path = "pretrained/VQVAE/net_last.pth"
    transformer_checkpoint_path: str | Path = "pretrained/VQTransformer_corruption05/net_best_fid.pth"
    mean_path: str | Path | None = None
    std_path: str | Path | None = None
    clip_model_name: str = "ViT-B/32"
    download_root: str | Path = "./"
    dataname: str = "t2m"
    device: str = "auto"
    nb_code: int = 512
    code_dim: int = 512
    output_emb_width: int = 512
    down_t: int = 2
    stride_t: int = 2
    width: int = 512
    depth: int = 3
    dilation_growth_rate: int = 3
    quantizer: str = "ema_reset"
    mu: float = 0.99
    quantbeta: float = 1.0
    block_size: int = 51
    clip_dim: int = 512
    transformer_embed_dim: int = 1024
    transformer_num_layers: int = 9
    transformer_num_heads: int = 16
    ff_rate: int = 4
    drop_out_rate: float = 0.1
    extra_mean_candidates: tuple[str, ...] = field(
        default_factory=lambda: (
            "dataset/HumanML3D/Mean.npy",
            "dataset/t2m/Mean.npy",
            "checkpoints/t2m/Mean.npy",
            "checkpoints/t2m/t2m/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy",
            "checkpoints/t2m/t2m/t2m/Comp_v6_KLD005/meta/mean.npy",
            "checkpoints/t2m/t2m/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy",
        )
    )
    extra_std_candidates: tuple[str, ...] = field(
        default_factory=lambda: (
            "dataset/HumanML3D/Std.npy",
            "dataset/t2m/Std.npy",
            "checkpoints/t2m/Std.npy",
            "checkpoints/t2m/t2m/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy",
            "checkpoints/t2m/t2m/t2m/Comp_v6_KLD005/meta/std.npy",
            "checkpoints/t2m/t2m/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy",
        )
    )


class LegacyT2MGPTBackend:
    """Thin wrapper around the repository's original T2M-GPT motion generation stack."""

    def __init__(self, config: LegacyT2MGPTBackendConfig | None = None) -> None:
        self.config = LegacyT2MGPTBackendConfig() if config is None else config
        self._runtime: Optional[Dict[str, Any]] = None

    def generate(
        self,
        *,
        prompt_text: str,
        seed: int,
        fps: float,
        duration_hint_sec: float | None = None,
        output_dir: str | Path | None = None,
    ) -> Dict[str, Any]:
        if prompt_text.strip() == "":
            raise ValueError("Prompt backend requires a non-empty prompt_text.")
        runtime = self._ensure_runtime()
        torch = runtime["torch"]

        _seed_everything(torch=torch, seed=int(seed))

        text_tokens = runtime["clip"].tokenize([str(prompt_text)], truncate=True).to(runtime["device"])
        with torch.no_grad():
            clip_features = runtime["clip_model"].encode_text(text_tokens).float()
            motion_indices = runtime["transformer"].sample(clip_features[0:1], False)
            decoded_motion = runtime["vqvae"].forward_decoder(motion_indices)
            predicted_xyz = runtime["recover_from_ric"](
                (decoded_motion * runtime["std_tensor"] + runtime["mean_tensor"]).float(),
                22,
            )

        joint_positions_xyz = predicted_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
        if joint_positions_xyz.ndim == 4:
            joint_positions_xyz = joint_positions_xyz[0]
        if output_dir is not None:
            output_dir = Path(output_dir)
        return {
            "joint_positions_xyz": joint_positions_xyz,
            "generation_backend": "t2mgpt",
            "backend_report": {
                "backend_name": "legacy_t2mgpt",
                "prompt_text": str(prompt_text),
                "seed": int(seed),
                "fps_requested": float(fps),
                "duration_hint_sec": (
                    None if duration_hint_sec is None else float(duration_hint_sec)
                ),
                "device": str(runtime["device"]),
                "vq_checkpoint_path": str(Path(self.config.vq_checkpoint_path).resolve()),
                "transformer_checkpoint_path": str(
                    Path(self.config.transformer_checkpoint_path).resolve()
                ),
                "mean_path": str(Path(runtime["mean_path"]).resolve()),
                "std_path": str(Path(runtime["std_path"]).resolve()),
                "output_dir": None if output_dir is None else str(output_dir.resolve()),
            },
            "artifacts": {},
        }

    def _ensure_runtime(self) -> Dict[str, Any]:
        if self._runtime is not None:
            return self._runtime

        try:
            import clip  # type: ignore
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover - import failure path
            raise RuntimeError(
                "Legacy T2M-GPT backend requires the local CLIP and torch dependencies."
            ) from exc

        import models.t2m_trans as trans
        import models.vqvae as vqvae
        from utils.motion_process import recover_from_ric

        device = _resolve_device(torch=torch, requested=str(self.config.device))
        mean_path = _resolve_stats_path(self.config.mean_path, self.config.extra_mean_candidates, label="mean")
        std_path = _resolve_stats_path(self.config.std_path, self.config.extra_std_candidates, label="std")
        args = _build_legacy_args(self.config)

        clip_model, _ = clip.load(
            self.config.clip_model_name,
            device=device,
            jit=False,
            download_root=str(self.config.download_root),
        )
        clip.model.convert_weights(clip_model)
        clip_model.eval()
        for parameter in clip_model.parameters():
            parameter.requires_grad = False

        vq_model = vqvae.HumanVQVAE(
            args,
            args.nb_code,
            args.code_dim,
            args.output_emb_width,
            args.down_t,
            args.stride_t,
            args.width,
            args.depth,
            args.dilation_growth_rate,
        )
        transformer = trans.Text2Motion_Transformer(
            num_vq=args.nb_code,
            embed_dim=args.transformer_embed_dim,
            clip_dim=args.clip_dim,
            block_size=args.block_size,
            num_layers=args.transformer_num_layers,
            n_head=args.transformer_num_heads,
            drop_out_rate=args.drop_out_rate,
            fc_rate=args.ff_rate,
        )
        vq_checkpoint = torch.load(self.config.vq_checkpoint_path, map_location="cpu")
        transformer_checkpoint = torch.load(self.config.transformer_checkpoint_path, map_location="cpu")
        vq_model.load_state_dict(vq_checkpoint["net"], strict=True)
        transformer.load_state_dict(transformer_checkpoint["trans"], strict=True)
        vq_model.eval().to(device)
        transformer.eval().to(device)

        mean_array = np.load(mean_path).astype(np.float32, copy=False)
        std_array = np.load(std_path).astype(np.float32, copy=False)
        self._runtime = {
            "clip": clip,
            "torch": torch,
            "device": device,
            "clip_model": clip_model,
            "vqvae": vq_model,
            "transformer": transformer,
            "recover_from_ric": recover_from_ric,
            "mean_tensor": torch.from_numpy(mean_array).to(device),
            "std_tensor": torch.from_numpy(std_array).to(device),
            "mean_path": str(mean_path),
            "std_path": str(std_path),
        }
        return self._runtime


def _build_legacy_args(config: LegacyT2MGPTBackendConfig) -> SimpleNamespace:
    return SimpleNamespace(
        dataname=str(config.dataname),
        nb_code=int(config.nb_code),
        code_dim=int(config.code_dim),
        output_emb_width=int(config.output_emb_width),
        down_t=int(config.down_t),
        stride_t=int(config.stride_t),
        width=int(config.width),
        depth=int(config.depth),
        dilation_growth_rate=int(config.dilation_growth_rate),
        quantizer=str(config.quantizer),
        mu=float(config.mu),
        quantbeta=float(config.quantbeta),
        block_size=int(config.block_size),
        clip_dim=int(config.clip_dim),
        ff_rate=int(config.ff_rate),
        drop_out_rate=float(config.drop_out_rate),
        transformer_embed_dim=int(config.transformer_embed_dim),
        transformer_num_layers=int(config.transformer_num_layers),
        transformer_num_heads=int(config.transformer_num_heads),
    )


def _resolve_stats_path(
    configured_path: str | Path | None,
    candidate_paths: tuple[str, ...],
    *,
    label: str,
) -> Path:
    candidates = []
    if configured_path not in (None, ""):
        candidates.append(Path(configured_path))
    candidates.extend(Path(candidate) for candidate in candidate_paths)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Legacy T2M-GPT backend could not resolve the {label} statistics file. "
        f"Checked: {[str(candidate) for candidate in candidates]}"
    )


def _resolve_device(*, torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(requested))


def _seed_everything(*, torch: Any, seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():  # pragma: no cover - depends on runtime hardware
        torch.cuda.manual_seed_all(int(seed))
