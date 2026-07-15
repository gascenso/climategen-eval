"""TensorFlow CNN callback for the leakage-audited GAUGE Track-2 ladder.

TensorFlow is imported lazily so the GAUGE core remains framework agnostic.
The residual multi-head architecture is adapted from ``tc-diffusion``'s
``tc_diffusion.evaluator`` implementation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from gauge_climate import Dataset
from gauge_climate.adapters.tc_data import wind_to_category
from gauge_climate.data import normalized_blocks
from gauge_climate.downstream import ArmBatch


@dataclass(frozen=True)
class TCCNNConfig:
    bt_min_k: float = 170.0
    bt_max_k: float = 325.0
    stage_channels: tuple[int, ...] = (32, 64, 128, 256)
    blocks_per_stage: int = 2
    embedding_dim: int = 256
    group_norm_groups: int = 8
    spatial_dropout_rate: float = 0.05
    embedding_dropout_rate: float = 0.25
    epochs: int = 27
    batch_size: int = 16
    learning_rate: float = 3.0e-4
    weight_decay: float = 5.0e-4
    huber_delta: float = 1.0
    label_smoothing: float = 0.04
    classification_weight: float = 1.0
    regression_weight: float = 1.0
    gradient_clip_norm: float | None = 1.0
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 5
    minimum_learning_rate: float = 1.0e-6
    early_stopping_patience: int = 6
    min_delta_kt: float = 1.0e-3
    deterministic_ops: bool = True

    def __post_init__(self) -> None:
        if not self.bt_min_k < self.bt_max_k:
            raise ValueError("bt_min_k must be below bt_max_k.")
        if not self.stage_channels or any(value <= 0 for value in self.stage_channels):
            raise ValueError("stage_channels must contain positive integers.")
        if self.group_norm_groups < 1 or any(
            value % self.group_norm_groups for value in self.stage_channels
        ):
            raise ValueError(
                "group_norm_groups must divide every stage channel count."
            )
        if self.blocks_per_stage < 1 or self.embedding_dim < 1:
            raise ValueError("CNN depth and embedding dimension must be positive.")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1).")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when provided.")
        if not 0 < self.reduce_lr_factor < 1:
            raise ValueError("reduce_lr_factor must be in (0, 1).")
        if self.reduce_lr_patience < 1 or self.minimum_learning_rate <= 0:
            raise ValueError("Learning-rate plateau settings must be positive.")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive.")

    def metadata(self) -> dict[str, Any]:
        return {
            "task": "TC Vmax regression with secondary native-category classification",
            "architecture": "tc_evaluator_resnet_multihead_v1",
            "source": "adapted from tc-diffusion/tc_diffusion/evaluator",
            "primary_endpoint": "storm-balanced Vmax MAE (kt)",
            "secondary_endpoint": "storm-balanced category accuracy",
            "wind_source": "USA_WIND",
            "wind_averaging_period": "1-minute",
            "training_budget": "restricted-real sample count per epoch for every arm",
            "validation": (
                "fixed storm-disjoint real validation split; checkpoint selected "
                "by storm-balanced Vmax MAE"
            ),
            "config": asdict(self),
        }


def tc_geometry(
    fields: np.ndarray, labels: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, Any]]:
    """Right-angle rotations plus reflect-padded shifts, applied to every added row."""

    del labels
    source = np.asarray(fields)
    if source.ndim not in (3, 4):
        raise ValueError("TC geometry expects (N,H,W) or (N,H,W,C) fields.")
    if source.shape[1] != source.shape[2]:
        raise ValueError("Quarter-turn TC geometry requires square fields.")
    output = np.empty_like(source)
    rotations: list[int] = []
    translations: list[list[int]] = []
    for index, sample in enumerate(source):
        rotation = int(rng.integers(0, 4))
        shift_y = int(rng.integers(-2, 3))
        shift_x = int(rng.integers(-2, 3))
        if rotation == 0 and shift_y == 0 and shift_x == 0:
            rotation = 1
        transformed = np.rot90(sample, k=rotation, axes=(0, 1))
        pad = ((2, 2), (2, 2)) if transformed.ndim == 2 else ((2, 2), (2, 2), (0, 0))
        transformed = np.pad(transformed, pad, mode="reflect")
        output[index] = transformed[
            2 + shift_y : 2 + shift_y + sample.shape[0],
            2 + shift_x : 2 + shift_x + sample.shape[1],
            ...,
        ]
        rotations.append(rotation)
        translations.append([shift_y, shift_x])
    return output, {
        "policy": "tc_rot90_reflect_shift_v1",
        "rotation_quarter_turns": rotations,
        "translation_pixels_yx": translations,
        "explicit_parameters": True,
    }


class TCCNNTask:
    """Callable that trains one paired TC evaluator and scores withheld storms."""

    def __init__(
        self,
        *,
        full_training_archive: Dataset,
        validation_archive: Dataset,
        holdout_blocks: Mapping[str, Sequence[str]],
        output_root: str | Path = "runs/tc_track2",
        config: TCCNNConfig = TCCNNConfig(),
    ) -> None:
        for name, dataset in {
            "full_training_archive": full_training_archive,
            "validation_archive": validation_archive,
        }.items():
            if dataset.blocks is None or dataset.labels is None:
                raise ValueError(f"{name} requires storm blocks and continuous wind labels.")
        self.full_training_archive = full_training_archive
        self.validation_archive = validation_archive
        self.holdout_blocks = {
            str(key): tuple(str(value) for value in values)
            for key, values in holdout_blocks.items()
        }
        if not self.holdout_blocks or any(not values for values in self.holdout_blocks.values()):
            raise ValueError("Every downstream holdout requires at least one storm block.")
        self.output_root = Path(output_root)
        self.config = config

    @property
    def metadata(self) -> dict[str, Any]:
        return self.config.metadata()

    def __call__(self, batch: ArmBatch, seed: int, holdout_id: str) -> Mapping[str, Any]:
        tf = _tensorflow()
        if holdout_id not in self.holdout_blocks:
            raise KeyError(f"Unknown downstream holdout {holdout_id!r}.")
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(int(seed))
        if self.config.deterministic_ops:
            tf.config.experimental.enable_op_determinism()

        base_count = int(batch.metadata.get("restricted_count", batch.labels.shape[0]))
        if not 1 < base_count <= batch.labels.shape[0]:
            raise ValueError("Arm metadata contains an invalid restricted_count.")
        base_wind = np.asarray(batch.labels[:base_count], dtype=np.float32)
        wind_mean = float(np.mean(base_wind))
        wind_std = float(np.std(base_wind))
        if not wind_std > 0:
            raise ValueError("Restricted real winds have zero variance.")

        sample_shape = tuple(int(value) for value in batch.fields.shape[1:])
        if len(sample_shape) == 2:
            input_shape = (*sample_shape, 1)
        elif len(sample_shape) == 3 and sample_shape[-1] == 1:
            input_shape = sample_shape
        else:
            raise ValueError(f"TC CNN expects scalar 2-D fields, got {sample_shape}.")
        model = _build_model(tf, input_shape=input_shape, config=self.config)
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        validation = _dataset_arrays(self.validation_archive)
        history: list[dict[str, float]] = []
        best_weights: list[np.ndarray] | None = None
        best_validation = np.inf
        epochs_without_improvement = 0
        epochs_without_lr_improvement = 0
        rng = np.random.default_rng(int(seed))
        for epoch in range(self.config.epochs):
            order = rng.permutation(batch.labels.shape[0])[:base_count]
            losses = []
            for indices in _batches(order, self.config.batch_size):
                x = _preprocess_fields(batch.fields[indices], self.config)
                wind = np.asarray(batch.labels[indices], dtype=np.float32)
                categories = wind_to_category(wind).astype(np.int32)
                wind_z = (wind - wind_mean) / wind_std
                weights = (
                    np.ones(indices.size, dtype=np.float32)
                    if batch.sample_weights is None
                    else np.asarray(batch.sample_weights[indices], dtype=np.float32)
                )
                with tf.GradientTape() as tape:
                    outputs = model(tf.convert_to_tensor(x), training=True)
                    if self.config.label_smoothing > 0:
                        category_targets = tf.one_hot(categories, depth=6)
                        smoothing = self.config.label_smoothing
                        category_targets = (
                            category_targets * (1.0 - smoothing) + smoothing / 6.0
                        )
                        classification = tf.nn.softmax_cross_entropy_with_logits(
                            labels=category_targets,
                            logits=outputs["class_logits"],
                        )
                    else:
                        classification = (
                            tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=categories,
                                logits=outputs["class_logits"],
                            )
                        )
                    error = tf.squeeze(outputs["wind_z"], axis=-1) - wind_z
                    absolute_error = tf.abs(error)
                    regression = tf.where(
                        absolute_error <= self.config.huber_delta,
                        0.5 * tf.square(error),
                        self.config.huber_delta
                        * (absolute_error - 0.5 * self.config.huber_delta),
                    )
                    classification_loss = tf.reduce_sum(
                        classification * weights
                    ) / tf.reduce_sum(weights)
                    regression_loss = tf.reduce_sum(
                        regression * weights
                    ) / tf.reduce_sum(
                        weights
                    )
                    loss = (
                        self.config.classification_weight * classification_loss
                        + self.config.regression_weight * regression_loss
                    )
                gradients = tape.gradient(loss, model.trainable_variables)
                if self.config.gradient_clip_norm is not None:
                    gradients, _ = tf.clip_by_global_norm(
                        gradients, self.config.gradient_clip_norm
                    )
                optimizer.apply_gradients(zip(gradients, model.trainable_variables, strict=True))
                losses.append(float(loss.numpy()))
            validation_prediction = _predict_wind(
                model,
                validation["fields"],
                self.config,
                wind_mean,
                wind_std,
                self.config.batch_size,
            )
            validation_mae = _block_balanced_mae(
                validation_prediction,
                validation["wind_kt"],
                validation["blocks"],
            )
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": float(np.mean(losses)),
                    "validation_vmax_mae_kt": validation_mae,
                }
            )
            if validation_mae < best_validation - self.config.min_delta_kt:
                best_validation = validation_mae
                best_weights = model.get_weights()
                epochs_without_improvement = 0
                epochs_without_lr_improvement = 0
            else:
                epochs_without_improvement += 1
                epochs_without_lr_improvement += 1
                if epochs_without_lr_improvement >= self.config.reduce_lr_patience:
                    current_rate = float(optimizer.learning_rate.numpy())
                    optimizer.learning_rate.assign(
                        max(
                            current_rate * self.config.reduce_lr_factor,
                            self.config.minimum_learning_rate,
                        )
                    )
                    epochs_without_lr_improvement = 0
                if epochs_without_improvement >= self.config.early_stopping_patience:
                    break
        if best_weights is None:
            raise RuntimeError("TC CNN training did not produce a finite validation checkpoint.")
        model.set_weights(best_weights)

        all_blocks = normalized_blocks(self.full_training_archive.blocks)
        requested = np.asarray(self.holdout_blocks[holdout_id], dtype=object)
        test_indices = np.flatnonzero(np.isin(all_blocks, requested))
        if test_indices.size == 0:
            raise ValueError(f"No frames found for downstream holdout {holdout_id!r}.")
        test_fields = self.full_training_archive.fields
        true_wind = np.asarray(self.full_training_archive.labels[test_indices], dtype=np.float32)
        predicted_wind, predicted_category = _predict_heads(
            model,
            test_fields,
            test_indices,
            self.config,
            wind_mean,
            wind_std,
            self.config.batch_size,
        )
        true_category = wind_to_category(true_wind)
        test_blocks = all_blocks[test_indices]
        metrics = {
            "vmax_mae_kt": {},
            "vmax_rmse_kt": {},
            "category_accuracy": {},
        }
        for block in self.holdout_blocks[holdout_id]:
            mask = test_blocks == block
            if not np.any(mask):
                raise ValueError(f"Holdout storm {block!r} has no evaluation frames.")
            error = predicted_wind[mask] - true_wind[mask]
            metrics["vmax_mae_kt"][block] = float(np.mean(np.abs(error)))
            metrics["vmax_rmse_kt"][block] = float(np.sqrt(np.mean(np.square(error))))
            metrics["category_accuracy"][block] = float(
                np.mean(predicted_category[mask] == true_category[mask])
            )

        run_dir = self.output_root / holdout_id / f"seed_{seed}" / batch.arm.value
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "protocol.json").write_text(
            json.dumps(
                {
                    "schema": "gauge_climate.tc_downstream_cnn.v1",
                    "holdout": holdout_id,
                    "storms": list(self.holdout_blocks[holdout_id]),
                    "seed": int(seed),
                    "arm": batch.arm.value,
                    "restricted_count": base_count,
                    "total_available_count": int(batch.labels.shape[0]),
                    "wind_standardization": {"mean_kt": wind_mean, "std_kt": wind_std},
                    "config": asdict(self.config),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        model.save_weights(run_dir / "best.weights.h5")
        return metrics


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "The TC downstream callback requires TensorFlow. Install "
            "`gauge-climate[tc-downstream]`."
        ) from exc
    return tf


def _build_model(tf, *, input_shape: tuple[int, ...], config: TCCNNConfig):
    layers = tf.keras.layers
    inputs = tf.keras.Input(shape=input_shape, name="bt")
    x = layers.Conv2D(config.stage_channels[0], 3, padding="same", use_bias=False)(inputs)
    x = layers.GroupNormalization(
        groups=config.group_norm_groups, epsilon=1.0e-5
    )(x)
    x = layers.Activation(tf.nn.silu)(x)
    if config.spatial_dropout_rate > 0:
        x = layers.SpatialDropout2D(config.spatial_dropout_rate)(x)
    for stage_index, channels in enumerate(config.stage_channels):
        for block_index in range(config.blocks_per_stage):
            stride = 2 if stage_index > 0 and block_index == 0 else 1
            shortcut = x
            h = layers.Conv2D(
                channels, 3, strides=stride, padding="same", use_bias=False
            )(x)
            h = layers.GroupNormalization(
                groups=config.group_norm_groups, epsilon=1.0e-5
            )(h)
            h = layers.Activation(tf.nn.silu)(h)
            h = layers.Conv2D(channels, 3, padding="same", use_bias=False)(h)
            h = layers.GroupNormalization(
                groups=config.group_norm_groups, epsilon=1.0e-5
            )(h)
            if stride != 1 or int(shortcut.shape[-1]) != channels:
                shortcut = layers.Conv2D(
                    channels, 1, strides=stride, padding="same", use_bias=False
                )(shortcut)
                shortcut = layers.GroupNormalization(
                    groups=config.group_norm_groups, epsilon=1.0e-5
                )(shortcut)
            x = layers.Activation(tf.nn.silu)(layers.Add()([shortcut, h]))
        if stage_index == 1 and config.spatial_dropout_rate > 0:
            x = layers.SpatialDropout2D(config.spatial_dropout_rate)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Activation(tf.nn.silu)(layers.Dense(config.embedding_dim)(x))
    if config.embedding_dropout_rate > 0:
        x = layers.Dropout(config.embedding_dropout_rate)(x)
    return tf.keras.Model(
        inputs,
        {
            "class_logits": layers.Dense(6, dtype="float32", name="class_logits")(x),
            "wind_z": layers.Dense(1, dtype="float32", name="wind_z")(x),
        },
        name="tc_evaluator_resnet",
    )


def _preprocess_fields(fields: Any, config: TCCNNConfig) -> np.ndarray:
    values = np.asarray(fields, dtype=np.float32)
    values = np.nan_to_num(
        values,
        nan=config.bt_max_k,
        posinf=config.bt_max_k,
        neginf=config.bt_min_k,
    )
    values = np.clip(values, config.bt_min_k, config.bt_max_k)
    values = 2.0 * (values - config.bt_min_k) / (config.bt_max_k - config.bt_min_k) - 1.0
    if values.ndim == 3:
        values = values[..., None]
    return values


def _batches(indices: np.ndarray, batch_size: int):
    for start in range(0, indices.size, batch_size):
        yield indices[start : start + batch_size]


def _dataset_arrays(dataset: Dataset) -> dict[str, Any]:
    return {
        "fields": dataset.fields,
        "wind_kt": np.asarray(dataset.labels, dtype=np.float32),
        "blocks": normalized_blocks(dataset.blocks),
    }


def _block_balanced_mae(
    prediction: np.ndarray,
    target: np.ndarray,
    blocks: np.ndarray,
) -> float:
    errors = np.abs(
        np.asarray(prediction, dtype=np.float64)
        - np.asarray(target, dtype=np.float64)
    )
    normalized = np.asarray(blocks, dtype=object)
    unique = list(dict.fromkeys(normalized.tolist()))
    return float(
        np.mean([np.mean(errors[normalized == block]) for block in unique])
    )


def _predict_wind(
    model,
    fields: Any,
    config: TCCNNConfig,
    wind_mean: float,
    wind_std: float,
    batch_size: int,
) -> np.ndarray:
    predictions = []
    for indices in _batches(np.arange(fields.shape[0]), batch_size):
        output = model(_preprocess_fields(fields[indices], config), training=False)
        predictions.append(np.asarray(output["wind_z"]).reshape(-1) * wind_std + wind_mean)
    return np.concatenate(predictions).astype(np.float32)


def _predict_heads(
    model,
    fields: Any,
    indices: np.ndarray,
    config: TCCNNConfig,
    wind_mean: float,
    wind_std: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    wind_predictions = []
    category_predictions = []
    for batch_indices in _batches(indices, batch_size):
        output = model(_preprocess_fields(fields[batch_indices], config), training=False)
        wind_predictions.append(
            np.asarray(output["wind_z"]).reshape(-1) * wind_std + wind_mean
        )
        category_predictions.append(
            np.argmax(np.asarray(output["class_logits"]), axis=-1).astype(np.int16)
        )
    return (
        np.concatenate(wind_predictions).astype(np.float32),
        np.concatenate(category_predictions),
    )
