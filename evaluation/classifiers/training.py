from __future__ import annotations

from dataclasses import dataclass
import copy
from itertools import cycle
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency guard
    tqdm = None

from .metrics import compute_multitask_metrics
from .model import MultitaskFusionClassifier, ensure_torch_available

_TORCH_IMPORT_ERROR: Exception | None = None
try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ImportError as exc:  # pragma: no cover - optional dependency guard
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = object  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]
    WeightedRandomSampler = object  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc


HEAD_TO_LOGIT_KEY = {
    "emotion": "emotion_logits",
    "modality": "modality_logits",
    "stimulus": "stimulus_logits",
    "flat_tag": "flat_tag_logits",
}


def ensure_training_dependencies() -> None:
    ensure_torch_available()
    if torch is None:
        raise ImportError(
            "evaluation.classifiers.training requires PyTorch."
        ) from _TORCH_IMPORT_ERROR


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    dropout: float = 0.1
    trunk_blocks: int = 2
    graph_layout: str = "imugpt22"
    use_flat_tag_head: bool = True
    use_domain_head: bool = True
    quality_dim: int = 8
    modality_dropout_p: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    classification_batch_size: int | None = None
    domain_batch_size: int | None = None
    max_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    grad_clip_norm: float = 1.0
    use_cb_focal: bool = False
    cb_focal_heads: tuple[str, ...] = ("emotion", "stimulus")
    focal_gamma: float = 2.0
    device: str = "auto"
    grl_lambda: float = 1.0
    domain_loss_weight: float = 0.1
    flat_tag_loss_weight: float = 0.0
    emotion_loss_weight: float = 1.0
    modality_loss_weight: float = 0.5
    stimulus_loss_weight: float = 1.0
    selection_head: str = "emotion"
    sampler_power: float = 1.0
    num_workers: int = 0
    show_progress: bool = True


class WindowTensorDataset(Dataset):
    def __init__(self, arrays: Mapping[str, Any]) -> None:
        self.pose = np.asarray(arrays["pose"], dtype=np.float32)
        self.imu = np.asarray(arrays["imu"], dtype=np.float32)
        self.quality = np.asarray(arrays["quality"], dtype=np.float32)
        self.domain = np.asarray(arrays["domain"], dtype=np.int64)
        self.classification_mask = np.asarray(arrays["classification_mask"], dtype=np.float32)
        self.targets = {
            head_name: np.asarray(values, dtype=np.int64)
            for head_name, values in dict(arrays["targets"]).items()
        }

    def __len__(self) -> int:
        return int(self.pose.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ensure_training_dependencies()
        item = {
            "pose": torch.from_numpy(self.pose[index]),
            "imu": torch.from_numpy(self.imu[index]),
            "quality": torch.from_numpy(self.quality[index]),
            "domain": torch.tensor(int(self.domain[index]), dtype=torch.long),
            "classification_mask": torch.tensor(float(self.classification_mask[index]), dtype=torch.float32),
        }
        for head_name, values in self.targets.items():
            item[f"{head_name}_target"] = torch.tensor(int(values[index]), dtype=torch.long)
        return item


def _resolve_device(device: str) -> torch.device:
    ensure_training_dependencies()
    normalized = str(device).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


def _normalize_pose_batch(pose_batch: torch.Tensor) -> torch.Tensor:
    mean = pose_batch.mean(dim=(1, 2), keepdim=True)
    std = pose_batch.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return (pose_batch - mean) / std


def _normalize_imu_batch(imu_batch: torch.Tensor) -> torch.Tensor:
    mean = imu_batch.mean(dim=1, keepdim=True)
    std = imu_batch.std(dim=1, keepdim=True).clamp_min(1e-6)
    return (imu_batch - mean) / std


def _compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    ensure_training_dependencies()
    present_classes = np.unique(np.asarray(labels, dtype=np.int64))
    if present_classes.size == 0:
        return torch.ones(num_classes, dtype=torch.float32)
    weights_present = compute_class_weight(class_weight="balanced", classes=present_classes, y=labels)
    weights = np.ones(int(num_classes), dtype=np.float32)
    weights[present_classes] = weights_present.astype(np.float32, copy=False)
    return torch.tensor(weights, dtype=torch.float32)


def _balanced_sample_weights(labels: np.ndarray, *, power: float = 1.0) -> np.ndarray:
    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.size == 0:
        return np.zeros(0, dtype=np.float32)
    counts = pd.Series(label_array).value_counts().to_dict()
    weights = np.ones(label_array.shape[0], dtype=np.float32)
    for sample_index, label in enumerate(label_array.tolist()):
        label_count = max(1, int(counts.get(int(label), 1)))
        weights[sample_index] = float((1.0 / label_count) ** float(power))
    return weights


def _weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.new_zeros(())
    num_classes = logits.shape[1]
    log_probs = torch.log_softmax(logits, dim=1)
    with torch.no_grad():
        target_distribution = torch.full_like(log_probs, fill_value=float(label_smoothing) / max(1, num_classes - 1))
        target_distribution.scatter_(1, targets[:, None], 1.0 - float(label_smoothing))
        if num_classes == 1:
            target_distribution.fill_(1.0)
    losses = -(target_distribution * log_probs).sum(dim=1)
    if class_weights is not None:
        losses = losses * class_weights.to(logits.device)[targets]
    return losses.mean()


def _class_balanced_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: torch.Tensor | None,
    gamma: float,
) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.new_zeros(())
    ce_loss = F.cross_entropy(logits, targets, weight=None, reduction="none")
    pt = torch.exp(-ce_loss)
    focal = torch.pow(1.0 - pt, float(gamma)) * ce_loss
    if class_weights is not None:
        focal = focal * class_weights.to(logits.device)[targets]
    return focal.mean()


def _head_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
    use_cb_focal: bool,
    focal_gamma: float,
) -> torch.Tensor:
    if use_cb_focal:
        return _class_balanced_focal_loss(
            logits,
            targets,
            class_weights=class_weights,
            gamma=focal_gamma,
        )
    return _weighted_cross_entropy(
        logits,
        targets,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
    )


def infer_model_shapes(arrays: Mapping[str, Any]) -> dict[str, int]:
    pose = np.asarray(arrays["pose"], dtype=np.float32)
    imu = np.asarray(arrays["imu"], dtype=np.float32)
    return {
        "pose_in_channels": int(pose.shape[-1]),
        "imu_in_channels": int(imu.shape[2] * imu.shape[3]),
        "quality_dim": int(np.asarray(arrays["quality"], dtype=np.float32).shape[1]),
    }


def _slice_arrays(arrays: Mapping[str, Any], indices: np.ndarray) -> dict[str, Any]:
    index_array = np.asarray(indices, dtype=np.int64)
    return {
        "pose": np.asarray(arrays["pose"], dtype=np.float32)[index_array],
        "imu": np.asarray(arrays["imu"], dtype=np.float32)[index_array],
        "quality": np.asarray(arrays["quality"], dtype=np.float32)[index_array],
        "domain": np.asarray(arrays["domain"], dtype=np.int64)[index_array],
        "classification_mask": np.asarray(arrays["classification_mask"], dtype=np.float32)[index_array],
        "targets": {
            head_name: np.asarray(values, dtype=np.int64)[index_array]
            for head_name, values in dict(arrays["targets"]).items()
        },
        "metadata": (
            arrays["metadata"].iloc[index_array].reset_index(drop=True)
            if "metadata" in arrays and arrays["metadata"] is not None
            else None
        ),
    }


def build_model(
    *,
    arrays: Mapping[str, Any],
    label_encoders: Mapping[str, Mapping[str, Any]],
    model_config: ModelConfig,
    use_pose_branch: bool,
    use_imu_branch: bool,
    use_domain_head: bool,
) -> MultitaskFusionClassifier:
    ensure_training_dependencies()
    shapes = infer_model_shapes(arrays)
    return MultitaskFusionClassifier(
        pose_in_channels=shapes["pose_in_channels"],
        imu_in_channels=shapes["imu_in_channels"],
        num_emotions=len(label_encoders["emotion"]["class_names"]),
        num_modalities=len(label_encoders["modality"]["class_names"]),
        num_stimuli=len(label_encoders["stimulus"]["class_names"]),
        num_flat_tags=(len(label_encoders["flat_tag"]["class_names"]) if model_config.use_flat_tag_head else None),
        use_pose_branch=bool(use_pose_branch),
        use_imu_branch=bool(use_imu_branch),
        use_domain_head=bool(use_domain_head and model_config.use_domain_head),
        graph_layout=str(model_config.graph_layout),
        hidden_dim=int(model_config.hidden_dim),
        trunk_blocks=int(model_config.trunk_blocks),
        dropout=float(model_config.dropout),
        quality_dim=int(shapes["quality_dim"] if model_config.quality_dim > 0 else 0),
        modality_dropout_p=float(model_config.modality_dropout_p),
    )


def _build_sampler(labels: np.ndarray, *, power: float) -> WeightedRandomSampler:
    ensure_training_dependencies()
    weights = _balanced_sample_weights(np.asarray(labels, dtype=np.int64), power=float(power))
    return WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights), replacement=True)


def _build_classification_arrays(arrays: Mapping[str, Any]) -> dict[str, Any]:
    supervised_indices = np.flatnonzero(np.asarray(arrays["classification_mask"], dtype=np.float32) > 0.0)
    if supervised_indices.size == 0:
        raise ValueError("Classification training requires at least one supervised sample.")
    return _slice_arrays(arrays, supervised_indices)


def _build_classification_loader(
    arrays: Mapping[str, Any],
    training_config: TrainingConfig,
) -> tuple[dict[str, Any], DataLoader]:
    classification_arrays = _build_classification_arrays(arrays)
    dataset = WindowTensorDataset(classification_arrays)
    sampler = _build_sampler(
        np.asarray(classification_arrays["targets"]["flat_tag"], dtype=np.int64),
        power=float(training_config.sampler_power),
    )
    batch_size = int(
        training_config.batch_size
        if training_config.classification_batch_size is None
        else training_config.classification_batch_size
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=int(training_config.num_workers),
    )
    return classification_arrays, loader


def _build_domain_loader(
    arrays: Mapping[str, Any],
    training_config: TrainingConfig,
) -> DataLoader:
    dataset = WindowTensorDataset(arrays)
    sampler = _build_sampler(
        np.asarray(arrays["domain"], dtype=np.int64),
        power=float(training_config.sampler_power),
    )
    batch_size = int(
        training_config.batch_size
        if training_config.domain_batch_size is None
        else training_config.domain_batch_size
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=int(training_config.num_workers),
    )


def _prepare_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    prepared = {key: value.to(device) for key, value in batch.items()}
    prepared["pose"] = _normalize_pose_batch(prepared["pose"])
    prepared["imu"] = _normalize_imu_batch(prepared["imu"])
    return prepared


def _gather_predictions(
    model: MultitaskFusionClassifier,
    arrays: Mapping[str, Any],
    label_encoders: Mapping[str, Mapping[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
    scored_class_ids: Mapping[str, Sequence[int]] | None = None,
    filter_unsupervised: bool = True,
) -> dict[str, Any]:
    ensure_training_dependencies()
    dataset = WindowTensorDataset(arrays)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    predictions: dict[str, list[np.ndarray]] = {head_name: [] for head_name in label_encoders.keys()}
    probabilities: dict[str, list[np.ndarray]] = {head_name: [] for head_name in label_encoders.keys()}
    targets: dict[str, np.ndarray] = {
        head_name: np.asarray(arrays["targets"][head_name], dtype=np.int64)
        for head_name in label_encoders.keys()
    }

    model.eval()
    with torch.no_grad():
        for batch in loader:
            prepared = _prepare_batch(batch, device)
            outputs = model(
                pose_inputs=prepared["pose"],
                imu_inputs=prepared["imu"],
                quality_inputs=prepared["quality"],
                domain_lambda=1.0,
            )
            for head_name in label_encoders.keys():
                logit_key = HEAD_TO_LOGIT_KEY[head_name]
                if logit_key not in outputs:
                    continue
                logits = outputs[logit_key]
                probs = torch.softmax(logits, dim=1)
                predictions[head_name].append(torch.argmax(probs, dim=1).cpu().numpy())
                probabilities[head_name].append(probs.cpu().numpy())

    merged_predictions = {
        head_name: np.concatenate(head_blocks, axis=0).astype(np.int64)
        for head_name, head_blocks in predictions.items()
        if len(head_blocks) > 0
    }
    merged_probabilities = {
        head_name: np.concatenate(head_blocks, axis=0).astype(np.float32)
        for head_name, head_blocks in probabilities.items()
        if len(head_blocks) > 0
    }
    supervised_mask = np.asarray(arrays["classification_mask"], dtype=np.float32) > 0.0
    evaluation_mask = supervised_mask if bool(filter_unsupervised) else np.ones_like(supervised_mask, dtype=bool)
    filtered_targets = {
        head_name: values[evaluation_mask]
        for head_name, values in targets.items()
        if head_name in merged_predictions
    }
    filtered_predictions = {
        head_name: values[evaluation_mask]
        for head_name, values in merged_predictions.items()
    }
    filtered_probabilities = {
        head_name: values[evaluation_mask]
        for head_name, values in merged_probabilities.items()
    }
    return {
        "targets": filtered_targets,
        "predictions": filtered_predictions,
        "probabilities": filtered_probabilities,
        "metadata": (
            arrays["metadata"].iloc[np.flatnonzero(evaluation_mask)].reset_index(drop=True)
            if "metadata" in arrays and arrays["metadata"] is not None
            else None
        ),
        "classification_mask": evaluation_mask.astype(bool),
        "metrics": compute_multitask_metrics(
            y_true=filtered_targets,
            y_pred=filtered_predictions,
            probabilities=filtered_probabilities,
            label_encoders=label_encoders,
            scored_class_ids=scored_class_ids,
        ),
    }


def evaluate_multitask_model(
    model: MultitaskFusionClassifier,
    arrays: Mapping[str, Any],
    label_encoders: Mapping[str, Mapping[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
    scored_class_ids: Mapping[str, Sequence[int]] | None = None,
    filter_unsupervised: bool = True,
) -> dict[str, Any]:
    ensure_training_dependencies()
    return _gather_predictions(
        model,
        arrays,
        label_encoders,
        batch_size=batch_size,
        device=device,
        scored_class_ids=scored_class_ids,
        filter_unsupervised=filter_unsupervised,
    )


def _resolve_selection_metric(
    report: Mapping[str, Any],
    *,
    selection_head: str,
) -> float:
    metrics = dict(report.get("metrics", {}))
    per_head = dict(metrics.get("per_head", {}))
    if selection_head in per_head:
        head_metrics = per_head[selection_head]
        return float(head_metrics.get("supported_macro_f1", head_metrics.get("macro_f1", 0.0)))
    global_metric = metrics.get("global_score_macro_f1_mean")
    return 0.0 if global_metric is None else float(global_metric)


def train_multitask_model(
    *,
    train_arrays: Mapping[str, Any],
    val_arrays: Mapping[str, Any],
    label_encoders: Mapping[str, Mapping[str, Any]],
    model_config: ModelConfig,
    training_config: TrainingConfig,
    use_pose_branch: bool,
    use_imu_branch: bool,
    use_domain_head: bool,
    scored_class_ids: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    ensure_training_dependencies()
    device = _resolve_device(training_config.device)
    model = build_model(
        arrays=train_arrays,
        label_encoders=label_encoders,
        model_config=model_config,
        use_pose_branch=use_pose_branch,
        use_imu_branch=use_imu_branch,
        use_domain_head=use_domain_head,
    ).to(device)

    classification_train_arrays, classification_loader = _build_classification_loader(train_arrays, training_config)
    domain_loader = _build_domain_loader(train_arrays, training_config) if bool(use_domain_head) else None
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.learning_rate),
        weight_decay=float(training_config.weight_decay),
    )

    class_weights = {
        head_name: _compute_class_weights(
            np.asarray(classification_train_arrays["targets"][head_name], dtype=np.int64),
            len(label_encoders[head_name]["class_names"]),
        ).to(device)
        for head_name in ("emotion", "modality", "stimulus", "flat_tag")
        if head_name in classification_train_arrays["targets"]
    }
    domain_class_weights = _compute_class_weights(
        np.asarray(train_arrays["domain"], dtype=np.int64),
        2,
    ).to(device)

    best_state = copy.deepcopy(model.state_dict())
    best_metric = -np.inf
    history: list[dict[str, Any]] = []

    epoch_indices = range(int(training_config.max_epochs))
    epoch_iterator = epoch_indices
    if bool(training_config.show_progress) and tqdm is not None:
        epoch_iterator = tqdm(epoch_indices, total=len(epoch_indices), desc="Training", unit="epoch")

    for epoch_index in epoch_iterator:
        model.train()
        running_loss = 0.0
        num_batches = 0
        domain_iterator = None if domain_loader is None else cycle(domain_loader)
        focal_heads = {str(head_name) for head_name in training_config.cb_focal_heads}
        batch_iterator = classification_loader
        if bool(training_config.show_progress) and tqdm is not None:
            batch_iterator = tqdm(
                classification_loader,
                total=len(classification_loader),
                desc=f"Epoch {int(epoch_index) + 1}/{int(training_config.max_epochs)}",
                unit="batch",
                leave=False,
            )

        for classification_batch in batch_iterator:
            prepared = _prepare_batch(classification_batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                pose_inputs=prepared["pose"],
                imu_inputs=prepared["imu"],
                quality_inputs=prepared["quality"],
                domain_lambda=0.0,
            )

            loss = torch.zeros((), device=device)
            for head_name, head_weight in (
                ("emotion", training_config.emotion_loss_weight),
                ("modality", training_config.modality_loss_weight),
                ("stimulus", training_config.stimulus_loss_weight),
            ):
                if float(head_weight) <= 0.0:
                    continue
                loss = loss + float(head_weight) * _head_loss(
                    outputs[HEAD_TO_LOGIT_KEY[head_name]],
                    prepared[f"{head_name}_target"],
                    class_weights=class_weights[head_name],
                    label_smoothing=float(training_config.label_smoothing),
                    use_cb_focal=bool(training_config.use_cb_focal and head_name in focal_heads),
                    focal_gamma=float(training_config.focal_gamma),
                )

            if "flat_tag_logits" in outputs and "flat_tag" in class_weights and float(training_config.flat_tag_loss_weight) > 0.0:
                loss = loss + float(training_config.flat_tag_loss_weight) * _head_loss(
                    outputs["flat_tag_logits"],
                    prepared["flat_tag_target"],
                    class_weights=class_weights["flat_tag"],
                    label_smoothing=float(training_config.label_smoothing),
                    use_cb_focal=bool(training_config.use_cb_focal and "flat_tag" in focal_heads),
                    focal_gamma=float(training_config.focal_gamma),
                )

            if bool(use_domain_head) and domain_iterator is not None:
                domain_batch = next(domain_iterator)
                prepared_domain = _prepare_batch(domain_batch, device)
                domain_outputs = model(
                    pose_inputs=prepared_domain["pose"],
                    imu_inputs=prepared_domain["imu"],
                    quality_inputs=prepared_domain["quality"],
                    domain_lambda=float(training_config.grl_lambda),
                )
                loss = loss + float(training_config.domain_loss_weight) * _head_loss(
                    domain_outputs["domain_logits"],
                    prepared_domain["domain"],
                    class_weights=domain_class_weights,
                    label_smoothing=0.0,
                    use_cb_focal=False,
                    focal_gamma=float(training_config.focal_gamma),
                )

            loss.backward()
            if float(training_config.grad_clip_norm) > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(training_config.grad_clip_norm))
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1
            if bool(training_config.show_progress) and tqdm is not None:
                batch_iterator.set_postfix(train_loss=f"{running_loss / max(1, num_batches):.4f}")

        if bool(training_config.show_progress) and tqdm is not None:
            batch_iterator.close()

        val_report = _gather_predictions(
            model,
            val_arrays,
            label_encoders,
            batch_size=int(training_config.batch_size),
            device=device,
            scored_class_ids=scored_class_ids,
            filter_unsupervised=True,
        )
        val_metric_value = _resolve_selection_metric(
            val_report,
            selection_head=str(training_config.selection_head),
        )
        if val_metric_value > best_metric:
            best_metric = val_metric_value
            best_state = copy.deepcopy(model.state_dict())

        history.append(
            {
                "epoch": int(epoch_index + 1),
                "train_loss": float(running_loss / max(1, num_batches)),
                "val_selection_metric": float(val_metric_value),
                "val_global_score_macro_f1_mean": float(val_report["metrics"].get("global_score_macro_f1_mean") or 0.0),
            }
        )
        if bool(training_config.show_progress) and tqdm is not None:
            epoch_iterator.set_postfix(
                train_loss=f"{running_loss / max(1, num_batches):.4f}",
                val_metric=f"{float(val_metric_value):.4f}",
            )

    if bool(training_config.show_progress) and tqdm is not None:
        epoch_iterator.close()

    model.load_state_dict(best_state)
    final_train_report = _gather_predictions(
        model,
        train_arrays,
        label_encoders,
        batch_size=int(training_config.batch_size),
        device=device,
        scored_class_ids=scored_class_ids,
        filter_unsupervised=True,
    )
    final_val_report = _gather_predictions(
        model,
        val_arrays,
        label_encoders,
        batch_size=int(training_config.batch_size),
        device=device,
        scored_class_ids=scored_class_ids,
        filter_unsupervised=True,
    )
    return {
        "model": model,
        "history": history,
        "train_report": final_train_report,
        "val_report": final_val_report,
        "best_val_selection_metric": None if best_metric == -np.inf else float(best_metric),
        "device": str(device),
    }
