from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data_loading import load_ml_inputs


def prepare_vae_inputs(config, logger):
    """Prepare scaled arrays, train/validation/test splits, and aligned handoff tables."""
    config.ensure_directories()

    logger.info("Loading ML inputs.")
    expression, metadata = load_ml_inputs(config, require_full_contract=True)

    logger.info("Expression matrix: %s samples x %s features.", expression.shape[0], expression.shape[1])
    logger.info("Metadata rows: %s.", metadata.shape[0])
    logger.info("Condition counts: %s", metadata["condition"].value_counts(dropna=False).to_dict())
    logger.info("Disease labels: %s.", metadata["disease_clean"].nunique())
    logger.info("Tissue labels: %s.", metadata["tissue_clean"].nunique())

    X = expression.to_numpy(dtype=np.float32)
    logger.info(
        "Raw expression summary: min=%.3f max=%.3f mean=%.3f sd=%.3f.",
        float(X.min()), float(X.max()), float(X.mean()), float(X.std()),
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    logger.info(
        "Scaled expression summary: mean=%.6f sd=%.6f.",
        float(X_scaled.mean()), float(X_scaled.std()),
    )

    sample_ids = expression.index.to_numpy(dtype=str)

    X_train, X_temp, ids_train, ids_temp = train_test_split(
        X_scaled,
        sample_ids,
        test_size=config.test_size_total,
        random_state=config.random_seed,
    )

    X_val, X_test, ids_val, ids_test = train_test_split(
        X_temp,
        ids_temp,
        test_size=config.val_fraction_of_temp,
        random_state=config.random_seed,
    )

    logger.info("Train shape: %s.", X_train.shape)
    logger.info("Validation shape: %s.", X_val.shape)
    logger.info("Test shape: %s.", X_test.shape)

    np.save(config.x_train_path, X_train)
    np.save(config.x_val_path, X_val)
    np.save(config.x_test_path, X_test)
    np.save(config.ids_train_path, ids_train)
    np.save(config.ids_val_path, ids_val)
    np.save(config.ids_test_path, ids_test)
    np.save(config.x_scaled_all_path, X_scaled)
    np.save(config.sample_ids_all_path, sample_ids)
    np.save(config.feature_ids_path, expression.columns.to_numpy(dtype=str))

    metadata.to_csv(config.metadata_aligned_path, index=False)
    expression.to_csv(config.expression_aligned_path, index=True)

    scaler_stats = pd.DataFrame(
        {
            "feature": expression.columns,
            "mean": scaler.mean_,
            "scale": scaler.scale_,
        }
    )
    scaler_stats.to_csv(config.scaler_stats_path, index=False)

    logger.info("Saved processed arrays to: %s", config.processed_dir)
    logger.info("Saved aligned metadata: %s", config.metadata_aligned_path)
    logger.info("Saved aligned expression: %s", config.expression_aligned_path)
    logger.info("Saved scaler statistics: %s", config.scaler_stats_path)

    return {
        "n_samples": int(expression.shape[0]),
        "n_features": int(expression.shape[1]),
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
    }
