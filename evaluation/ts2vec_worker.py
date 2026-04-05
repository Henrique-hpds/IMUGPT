from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - depends on sklearn version
    StratifiedGroupKFold = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TS2VEC_ROOT = PROJECT_ROOT / "ts2vec"
if str(TS2VEC_ROOT) not in sys.path:
    sys.path.insert(0, str(TS2VEC_ROOT))

from ts2vec import TS2Vec  # noqa: E402


class SequenceGRUPredictor(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        effective_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=effective_dropout,
            batch_first=True,
        )
        self.head = torch.nn.Linear(int(hidden_size), int(input_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return self.head(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TS2Vec embeddings and predictive score in the ts2vec conda env.")
    parser.add_argument("--input-npz", required=True, help="Path to input NPZ.")
    parser.add_argument("--groups-json", required=True, help="Path to metric-group JSON definition.")
    parser.add_argument("--output-npz", required=True, help="Path to output NPZ.")
    parser.add_argument("--repr-dims", type=int, default=64)
    parser.add_argument("--hidden-dims", type=int, default=64)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-iters", type=int, default=200)
    parser.add_argument("--temporal-unit", type=int, default=0)
    parser.add_argument("--ts2vec-device", default="cpu")
    parser.add_argument("--gru-hidden-size", type=int, default=64)
    parser.add_argument("--gru-num-layers", type=int, default=1)
    parser.add_argument("--gru-dropout", type=float, default=0.0)
    parser.add_argument("--gru-batch-size", type=int, default=64)
    parser.add_argument("--gru-epochs", type=int, default=20)
    parser.add_argument("--gru-learning-rate", type=float, default=1e-3)
    parser.add_argument("--gru-weight-decay", type=float, default=1e-4)
    parser.add_argument("--gru-patience", type=int, default=4)
    parser.add_argument("--cv-folds", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def resolve_device(device_name: str) -> str:
    normalized = str(device_name).strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return normalized


def load_worker_payload(input_npz: str | Path) -> dict[str, Any]:
    with np.load(Path(input_npz), allow_pickle=False) as payload:
        return {
            "embedder_train_windows": np.asarray(payload["embedder_train_windows"], dtype=np.float32),
            "eval_real_windows": np.asarray(payload["eval_real_windows"], dtype=np.float32),
            "eval_synthetic_windows": np.asarray(payload["eval_synthetic_windows"], dtype=np.float32),
            "capture_ids": np.asarray(payload["capture_ids"]).astype(str),
        }


def fit_ts2vec_embeddings(
    train_windows: np.ndarray,
    eval_real_windows: np.ndarray,
    eval_synthetic_windows: np.ndarray,
    *,
    repr_dims: int,
    hidden_dims: int,
    depth: int,
    batch_size: int,
    n_iters: int,
    temporal_unit: int,
    device: str,
) -> dict[str, Any]:
    if train_windows.ndim != 3:
        raise ValueError("embedder_train_windows must have shape [n_windows, time, features].")
    if train_windows.shape[0] < 2:
        raise ValueError("TS2Vec training needs at least 2 real windows.")

    model = TS2Vec(
        input_dims=int(train_windows.shape[-1]),
        output_dims=int(repr_dims),
        hidden_dims=int(hidden_dims),
        depth=int(depth),
        device=device,
        batch_size=int(batch_size),
        temporal_unit=int(temporal_unit),
    )
    loss_log = model.fit(train_windows, n_iters=int(n_iters), verbose=True)
    real_embeddings = model.encode(eval_real_windows, encoding_window="full_series")
    synthetic_embeddings = model.encode(eval_synthetic_windows, encoding_window="full_series")

    return {
        "loss_log": np.asarray(loss_log, dtype=np.float32),
        "real_embeddings": np.asarray(real_embeddings, dtype=np.float32),
        "synthetic_embeddings": np.asarray(synthetic_embeddings, dtype=np.float32),
    }


def _normalize_windows(
    train_windows: np.ndarray,
    *other_windows: np.ndarray,
) -> tuple[np.ndarray, ...]:
    mean = train_windows.mean(axis=(0, 1), keepdims=True)
    std = train_windows.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    normalized = [((train_windows - mean) / std).astype(np.float32)]
    normalized.extend([((window - mean) / std).astype(np.float32) for window in other_windows])
    return tuple(normalized)


def _make_sequence_io(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if windows.shape[1] < 2:
        raise ValueError("Predictive score requires windows with at least 2 timesteps.")
    inputs = np.asarray(windows[:, :-1, :], dtype=np.float32)
    targets = np.asarray(windows[:, 1:, :], dtype=np.float32)
    return inputs, targets


def _build_group_folds(
    num_samples: int,
    capture_ids: np.ndarray,
    requested_splits: int,
    random_state: int,
) -> tuple[str, list[tuple[np.ndarray, np.ndarray]]]:
    if num_samples < 2:
        return "unavailable", []

    requested_splits = max(2, int(requested_splits))
    unique_groups = np.unique(capture_ids)
    if unique_groups.size >= 2:
        n_splits = min(requested_splits, int(unique_groups.size))
        if StratifiedGroupKFold is not None and n_splits >= 2:
            pseudo_labels = np.arange(num_samples) % max(2, min(n_splits, 3))
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
            splits = list(splitter.split(np.zeros(num_samples), pseudo_labels, groups=capture_ids))
            return "group", [(np.asarray(train_idx), np.asarray(test_idx)) for train_idx, test_idx in splits]

        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(np.zeros(num_samples), groups=capture_ids))
        return "group", [(np.asarray(train_idx), np.asarray(test_idx)) for train_idx, test_idx in splits]

    n_splits = min(requested_splits, int(num_samples))
    if n_splits < 2:
        return "unavailable", []
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
    splits = list(splitter.split(np.zeros(num_samples)))
    return "window", [(np.asarray(train_idx), np.asarray(test_idx)) for train_idx, test_idx in splits]


def _train_gru_fold(
    train_windows: np.ndarray,
    test_windows: np.ndarray,
    *,
    device: str,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    normalized_train_windows, normalized_test_windows = _normalize_windows(train_windows, test_windows)
    train_inputs, train_targets = _make_sequence_io(normalized_train_windows)
    test_inputs, test_targets = _make_sequence_io(normalized_test_windows)

    if train_inputs.shape[0] < 2:
        raise ValueError("Each predictive-score fold needs at least 2 training windows.")

    permutation = np.random.default_rng(int(seed)).permutation(train_inputs.shape[0])
    val_size = max(1, int(round(train_inputs.shape[0] * 0.1))) if train_inputs.shape[0] >= 10 else 0

    val_idx = permutation[:val_size] if val_size > 0 else np.asarray([], dtype=np.int64)
    fit_idx = permutation[val_size:] if val_size > 0 else permutation
    if fit_idx.size == 0:
        fit_idx = permutation
        val_idx = np.asarray([], dtype=np.int64)

    fit_inputs = torch.from_numpy(train_inputs[fit_idx]).to(torch.float32)
    fit_targets = torch.from_numpy(train_targets[fit_idx]).to(torch.float32)
    val_inputs = None if val_idx.size == 0 else torch.from_numpy(train_inputs[val_idx]).to(torch.float32)
    val_targets = None if val_idx.size == 0 else torch.from_numpy(train_targets[val_idx]).to(torch.float32)

    model = SequenceGRUPredictor(
        input_size=int(train_inputs.shape[-1]),
        hidden_size=int(hidden_size),
        num_layers=int(num_layers),
        dropout=float(dropout),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    criterion = torch.nn.MSELoss()

    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(int(epochs)):
        model.train()
        epoch_losses = []
        fit_order = torch.randperm(fit_inputs.shape[0], device=fit_inputs.device)

        for start_idx in range(0, fit_inputs.shape[0], int(batch_size)):
            batch_idx = fit_order[start_idx : start_idx + int(batch_size)]
            batch_inputs = fit_inputs[batch_idx].to(device)
            batch_targets = fit_targets[batch_idx].to(device)

            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        if val_inputs is None:
            mean_train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            if mean_train_loss < best_val_loss:
                best_val_loss = mean_train_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_epoch = epoch
            continue

        model.eval()
        with torch.no_grad():
            val_predictions = model(val_inputs.to(device))
            val_loss = float(criterion(val_predictions, val_targets.to(device)).detach().cpu().item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch

        if (epoch - best_epoch) >= int(patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_predictions = model(torch.from_numpy(test_inputs).to(torch.float32).to(device)).cpu().numpy()

    flattened_target = test_targets.reshape(-1, test_targets.shape[-1])
    flattened_prediction = test_predictions.reshape(-1, test_predictions.shape[-1])

    r2 = float(r2_score(flattened_target, flattened_prediction, multioutput="uniform_average"))
    mae = float(mean_absolute_error(flattened_target, flattened_prediction))
    return {
        "r2": r2,
        "mae": mae,
    }


def compute_predictive_score(
    synthetic_windows: np.ndarray,
    real_windows: np.ndarray,
    capture_ids: np.ndarray,
    *,
    device: str,
    cv_folds: int,
    random_state: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
) -> dict[str, Any]:
    num_samples = int(min(synthetic_windows.shape[0], real_windows.shape[0]))
    if num_samples < 2:
        return {
            "mean": None,
            "std": None,
            "mae_mean": None,
            "mae_std": None,
            "cv_strategy": "unavailable",
            "num_splits": 0,
            "fold_scores": [],
            "fold_mae": [],
            "warning": "At least 2 windows are required for predictive score.",
            "metric_name": "r2",
        }

    syn = np.asarray(synthetic_windows[:num_samples], dtype=np.float32)
    real = np.asarray(real_windows[:num_samples], dtype=np.float32)
    groups = np.asarray(capture_ids[:num_samples]).astype(str)

    strategy, splits = _build_group_folds(
        num_samples=num_samples,
        capture_ids=groups,
        requested_splits=int(cv_folds),
        random_state=int(random_state),
    )
    if not splits:
        return {
            "mean": None,
            "std": None,
            "mae_mean": None,
            "mae_std": None,
            "cv_strategy": strategy,
            "num_splits": 0,
            "fold_scores": [],
            "fold_mae": [],
            "warning": "Could not build cross-validation folds for predictive score.",
            "metric_name": "r2",
        }

    fold_scores: list[float] = []
    fold_mae: list[float] = []
    for fold_index, (train_idx, test_idx) in enumerate(splits):
        if train_idx.size < 2 or test_idx.size < 1:
            continue
        metrics = _train_gru_fold(
            train_windows=syn[train_idx],
            test_windows=real[test_idx],
            device=device,
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=float(dropout),
            batch_size=int(batch_size),
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            patience=int(patience),
            seed=int(random_state) + fold_index,
        )
        fold_scores.append(float(metrics["r2"]))
        fold_mae.append(float(metrics["mae"]))

    if not fold_scores:
        return {
            "mean": None,
            "std": None,
            "mae_mean": None,
            "mae_std": None,
            "cv_strategy": strategy,
            "num_splits": 0,
            "fold_scores": [],
            "fold_mae": [],
            "warning": "All predictive-score folds were discarded because of insufficient data.",
            "metric_name": "r2",
        }

    ddof = 1 if len(fold_scores) > 1 else 0
    return {
        "mean": float(np.mean(fold_scores)),
        "std": float(np.std(fold_scores, ddof=ddof)),
        "mae_mean": float(np.mean(fold_mae)),
        "mae_std": float(np.std(fold_mae, ddof=ddof)),
        "cv_strategy": strategy,
        "num_splits": int(len(fold_scores)),
        "fold_scores": [float(score) for score in fold_scores],
        "fold_mae": [float(score) for score in fold_mae],
        "warning": None,
        "metric_name": "r2",
    }


def main() -> None:
    args = parse_args()
    payload = load_worker_payload(args.input_npz)
    groups = json.loads(Path(args.groups_json).read_text(encoding="utf-8"))
    device = resolve_device(args.ts2vec_device)

    embedding_result = fit_ts2vec_embeddings(
        payload["embedder_train_windows"],
        payload["eval_real_windows"],
        payload["eval_synthetic_windows"],
        repr_dims=int(args.repr_dims),
        hidden_dims=int(args.hidden_dims),
        depth=int(args.depth),
        batch_size=int(args.batch_size),
        n_iters=int(args.n_iters),
        temporal_unit=int(args.temporal_unit),
        device=device,
    )

    predictive_results: dict[str, Any] = {}
    for group in groups:
        group_key = str(group["group_key"])
        indices = np.asarray(group["indices"], dtype=np.int64)
        predictive_results[group_key] = compute_predictive_score(
            synthetic_windows=payload["eval_synthetic_windows"][indices],
            real_windows=payload["eval_real_windows"][indices],
            capture_ids=payload["capture_ids"][indices],
            device=device,
            cv_folds=int(args.cv_folds),
            random_state=int(args.random_state),
            hidden_size=int(args.gru_hidden_size),
            num_layers=int(args.gru_num_layers),
            dropout=float(args.gru_dropout),
            batch_size=int(args.gru_batch_size),
            epochs=int(args.gru_epochs),
            learning_rate=float(args.gru_learning_rate),
            weight_decay=float(args.gru_weight_decay),
            patience=int(args.gru_patience),
        )

    summary = {
        "ts2vec_device": device,
        "repr_dims": int(args.repr_dims),
        "train_windows": int(payload["embedder_train_windows"].shape[0]),
        "eval_windows": int(payload["eval_real_windows"].shape[0]),
    }

    np.savez_compressed(
        Path(args.output_npz),
        real_embeddings=np.asarray(embedding_result["real_embeddings"], dtype=np.float32),
        synthetic_embeddings=np.asarray(embedding_result["synthetic_embeddings"], dtype=np.float32),
        loss_log=np.asarray(embedding_result["loss_log"], dtype=np.float32),
        predictive_results_json=np.asarray(json.dumps(predictive_results), dtype=f"<U{max(1, len(json.dumps(predictive_results)))}"),
        worker_summary_json=np.asarray(json.dumps(summary), dtype=f"<U{max(1, len(json.dumps(summary)))}"),
    )


if __name__ == "__main__":
    main()
