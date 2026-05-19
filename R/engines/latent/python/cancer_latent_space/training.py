from __future__ import annotations

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .vae_model import VAE, vae_loss


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(array: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.tensor(array, dtype=torch.float32)
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle)


def train_vae(config, logger):
    """Train the VAE and save latent-space/model artifacts."""
    config.ensure_directories()
    set_random_seeds(config.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("PyTorch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())

    required = [
        config.x_train_path,
        config.x_val_path,
        config.x_test_path,
        config.x_scaled_all_path,
        config.sample_ids_all_path,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required prepared input not found: {path}")

    X_train = np.load(config.x_train_path)
    X_val = np.load(config.x_val_path)
    X_test = np.load(config.x_test_path)

    logger.info("Train shape: %s", X_train.shape)
    logger.info("Validation shape: %s", X_val.shape)
    logger.info("Test shape: %s", X_test.shape)

    train_loader = _make_loader(X_train, config.batch_size, shuffle=True)
    val_loader = _make_loader(X_val, config.batch_size, shuffle=False)

    input_dim = int(X_train.shape[1])
    latent_dim = int(config.latent_dim)

    model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    rows = []

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0

        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device)

            optimizer.zero_grad()
            x_recon, mu, logvar = model(x_batch)
            loss, recon_loss, kl_loss = vae_loss(
                x_batch, x_recon, mu, logvar, beta=config.beta
            )
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            train_recon += float(recon_loss.item())
            train_kl += float(kl_loss.item())

        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0

        with torch.no_grad():
            for (x_batch,) in val_loader:
                x_batch = x_batch.to(device)
                x_recon, mu, logvar = model(x_batch)
                loss, recon_loss, kl_loss = vae_loss(
                    x_batch, x_recon, mu, logvar, beta=config.beta
                )
                val_loss += float(loss.item())
                val_recon += float(recon_loss.item())
                val_kl += float(kl_loss.item())

        row = {
            "epoch": epoch + 1,
            "train_total_loss": train_loss,
            "train_recon_loss": train_recon,
            "train_kl_loss": train_kl,
            "val_total_loss": val_loss,
            "val_recon_loss": val_recon,
            "val_kl_loss": val_kl,
        }
        rows.append(row)

        logger.info(
            "Epoch %03d | Train total %.2f | Train recon %.2f | Train KL %.2f | "
            "Val total %.2f | Val recon %.2f | Val KL %.2f",
            epoch + 1,
            train_loss,
            train_recon,
            train_kl,
            val_loss,
            val_recon,
            val_kl,
        )

    history = pd.DataFrame(rows)
    history.to_csv(config.training_history_path, index=False)

    X_all = np.load(config.x_scaled_all_path)
    sample_ids = np.load(config.sample_ids_all_path, allow_pickle=True).astype(str)
    X_all_tensor = torch.tensor(X_all, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        mu, _logvar = model.encode(X_all_tensor)

    latent = mu.cpu().numpy()
    np.save(config.latent_path, latent)

    latent_df = pd.DataFrame(latent, columns=[f"z{i + 1}" for i in range(latent.shape[1])])
    latent_df.insert(0, "sample_id", sample_ids)
    latent_df.to_csv(config.latent_coordinates_path, index=False)

    torch.save(model.state_dict(), config.model_state_dict_path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "latent_dim": latent_dim,
            "input_dim": input_dim,
            "beta": config.beta,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "random_seed": config.random_seed,
        },
        config.model_checkpoint_path,
    )

    summary = pd.DataFrame(
        [
            {
                "input_dim": input_dim,
                "latent_dim": latent_dim,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "beta": config.beta,
                "learning_rate": config.learning_rate,
                "random_seed": config.random_seed,
                "final_train_total_loss": history["train_total_loss"].iloc[-1],
                "final_val_total_loss": history["val_total_loss"].iloc[-1],
                "latent_mean": float(latent.mean()),
                "latent_std": float(latent.std()),
            }
        ]
    )
    summary.to_csv(config.training_summary_path, index=False)

    if config.make_plots:
        save_training_plots(history, config.training_plots_dir)
        save_latent_histogram(latent, config.latent_plots_dir)

    logger.info("Saved training history: %s", config.training_history_path)
    logger.info("Saved training summary: %s", config.training_summary_path)
    logger.info("Saved latent array: %s", config.latent_path)
    logger.info("Saved latent coordinate table: %s", config.latent_coordinates_path)
    logger.info("Saved model state_dict: %s", config.model_state_dict_path)
    logger.info("Saved full checkpoint: %s", config.model_checkpoint_path)

    return summary.iloc[0].to_dict()


def save_training_plots(history: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_total_loss"], label="Train Total")
    plt.plot(history["epoch"], history["val_total_loss"], label="Validation Total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("VAE Training Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "vae_training_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(history["epoch"], history["train_recon_loss"], label="Train Recon")
    plt.plot(history["epoch"], history["val_recon_loss"], label="Val Recon")
    plt.plot(history["epoch"], history["train_kl_loss"], label="Train KL")
    plt.plot(history["epoch"], history["val_kl_loss"], label="Val KL")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Component")
    plt.legend()
    plt.title("VAE Loss Components")
    plt.tight_layout()
    plt.savefig(output_dir / "vae_loss_components.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_latent_histogram(latent: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(latent.ravel(), bins=50)
    plt.xlabel("Latent value")
    plt.ylabel("Frequency")
    plt.title("Distribution of latent variables")
    plt.tight_layout()
    plt.savefig(output_dir / "latent_value_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()
