from cancer_latent_space.training import train_vae
from cancer_latent_space.logging_utils import configure_logger
from cancer_latent_space.latent_config import LatentRuntimeConfig

import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--chip-id", required=True)
    parser.add_argument("--filter-regime", required=True)
    parser.add_argument("--top-n", type=int, required=True)

    parser.add_argument("--latent-dim", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)

    args = parser.parse_args()

    logger = configure_logger()

    config = LatentRuntimeConfig(
        project_dir=args.project_dir,
        chip_id=args.chip_id,
        filter_regime=args.filter_regime,
        top_n=args.top_n,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        beta=args.beta,
        learning_rate=args.learning_rate,
    )

    train_vae(config=config, logger=logger)


if __name__ == "__main__":
    main()
