import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import shap
from colorlog import ColoredFormatter
from sklearn import neural_network, svm

logger = logging.getLogger(__name__)


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


class ExplainerType(str, Enum):
    linear = "linear"
    kernel = "kernel"


def save_shap_value(X_test, y_test, shap_values, save_prefix, multi_class=False):
    save_path = Path(save_prefix) / Path("shap_explanation.csv")

    if multi_class:
        n_classes = len(shap_values)
        n_samples, n_features = shap_values[0].shape
        feature_names = X_test.columns

        shap_dfs = []

        for class_idx in range(n_classes):
            shap_df = pd.DataFrame(
                shap_values[class_idx],
                columns=[
                    f"SHAP_class{class_idx}_{feature}" for feature in feature_names
                ],
            )
            shap_dfs.append(shap_df)

        shap_all_classes_df = pd.concat(shap_dfs, axis=1)

        result_df = pd.concat(
            [
                X_test.reset_index(drop=True),
                pd.DataFrame({"Prediction": y_test}).reset_index(drop=True),
                shap_all_classes_df.reset_index(drop=True),
            ],
            axis=1,
        )

        result_df.to_csv(save_path, index=False)
    else:
        shap_df = pd.DataFrame(
            shap_values, columns=[f"SHAP_{col}" for col in X_test.columns]
        )

        result_df = pd.concat(
            [
                X_test.reset_index(drop=True),
                pd.DataFrame({"Prediction": y_test}).reset_index(drop=True),
                shap_df.reset_index(drop=True),
            ],
            axis=1,
        )

        result_df.to_csv(save_path, index=False)


def explain_model(model, X_train, X_test, y_train, y_test, explainer_type, save_prefix):

    is_classification = (
        isinstance(model, svm.SVC)
        or isinstance(model, svm.NuSVC)
        or isinstance(model, svm.LinearSVC)
        or isinstance(model, neural_network.MLPClassifier)
    )
    n_classes = pd.Series(y_train).nunique()

    multi_class = False
    if not is_classification:
        task_type = "regression"
    elif n_classes == 2:
        task_type = "binary"
    else:
        task_type = "multi_class"
        multi_class = True

    logger.info(f"Detected task type: {task_type}")

    if explainer_type == ExplainerType.kernel:
        explainer = shap.KernelExplainer(model.predict, X_train)
    elif explainer_type == ExplainerType.linear:
        explainer = shap.LinearExplainer(
            model, X_train, feature_perturbation="correlation_dependent"
        )

    shap_values = explainer.shap_values(X_test)

    if save_prefix is not None:
        save_shap_value(X_test, y_test, shap_values, save_prefix, multi_class)

    shap.summary_plot(shap_values, X_test)
    if save_prefix is not None:
        plt.savefig(Path(save_prefix) / Path("summary_plot.png"), bbox_inches="tight")

    shap.force_plot(
        explainer.expected_value, shap_values[0, :], X_test.iloc[0, :], matplotlib=True
    )
    if save_prefix is not None:
        plt.savefig(Path(save_prefix) / Path("force_plot.png"), bbox_inches="tight")
