import logging


def setup_logging(level: int = logging.INFO) -> None:
    if len(logging.getLogger().handlers) == 0:
        logging.basicConfig(
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            level=level,
        )


