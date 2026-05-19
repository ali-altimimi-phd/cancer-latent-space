from __future__ import annotations

from cancer_latent_space.latent_config import LatentRuntimeConfig
from cancer_latent_space.latent_tables import build_and_save_latent_working_table
from cancer_latent_space.logging_utils import configure_logger


def main() -> None:
    logger = configure_logger("latent_tables")
    config = LatentRuntimeConfig.from_project_root()
    build_and_save_latent_working_table(config, logger=logger)


if __name__ == "__main__":
    main()
