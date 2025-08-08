import logging
from typing import Optional

import pandas as pd
from colorlog import ColoredFormatter


def setup_logger(
    output_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:

    logger = logging.getLogger()
    logger.setLevel(level)

    if logger.handlers:
        return logger

    color_format = (
        "%(log_color)s[%(asctime)s] [%(levelname)s] [%(name)s]%(reset)s %(message)s"
    )
    color_formatter = ColoredFormatter(
        color_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
        secondary_log_colors={},
        style="%",
    )

    plain_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    plain_formatter = logging.Formatter(plain_format, datefmt="%Y-%m-%d %H:%M:%S")

    if output_file is not None:
        file_handler = logging.FileHandler(output_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)

    return logger


def load_data(path: str, label_column: Optional[str] = None):
    df = pd.read_csv(path)
    if label_column is None:
        return df

    X = df.drop(columns=[label_column])
    y = df[label_column]
    return X, y
