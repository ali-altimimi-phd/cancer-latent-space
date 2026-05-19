from cancer_latent_space.preprocessing import prepare_vae_inputs
from cancer_latent_space.logging_utils import configure_logger
from cancer_latent_space.latent_config import LatentRuntimeConfig

import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--chip-id", required=True)
    parser.add_argument("--filter-regime", required=True)
    parser.add_argument("--top-n", type=int, required=True)

    args = parser.parse_args()

    logger = configure_logger()

    config = LatentRuntimeConfig(
        project_dir=args.project_dir,
        chip_id=args.chip_id,
        filter_regime=args.filter_regime,
        top_n=args.top_n,
    )

    prepare_vae_inputs(config=config, logger=logger)


if __name__ == "__main__":
    main()
