import ast
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, get_args, get_origin

import typer
from sklearn import svm

from .utils import ExplainerType, explain_model, load_data

svm_app = typer.Typer()
logger = logging.getLogger(__name__)
UNSET = object()


class SVMType(str, Enum):
    C = "C"
    Nu = "Nu"
    Linear = "Linear"


class Kernel(str, Enum):
    linear = "linear"
    poly = "poly"
    rbf = "rbf"
    sigmoid = "sigmoid"
    precomputed = "precomputed"


class DecisionFunctionShape(str, Enum):
    ovo = "ovo"
    ovr = "ovr"


class Penalty(str, Enum):
    l1 = "l1"
    l2 = "l2"


class Loss(str, Enum):
    hinge = "hinge"
    squared_hinge = "squared_hinge"


class RegressionLoss(str, Enum):
    epsilon_insensitive = "epsilon_insensitive"
    squared_epsilon_insensitive = "squared_epsilon_insensitive"


class MultiClass(str, Enum):
    ovr = "ovr"
    crammer_singer = "crammer_singer"


@dataclass
class Config:
    C: float = 1.0
    nu: float = 0.5
    penalty: Penalty = Penalty.l2
    loss: Loss = Loss.squared_hinge
    dual: Union[str, bool] = "auto"
    multi_class: MultiClass = MultiClass.ovr
    fit_intercept: bool = True
    intercept_scaling: float = 1.0
    kernel: Kernel = Kernel.rbf
    degree: int = 3
    gamma: Union[float, str] = "scale"
    coef0: float = 0.0
    shrinking: bool = True
    probability: bool = False
    tol: float = 1e-3
    cache_size: float = 200
    class_weight: Optional[Union[dict, str]] = None
    verbose: Union[bool, int] = False
    max_iter: int = -1
    decision_function_shape: DecisionFunctionShape = DecisionFunctionShape.ovr
    break_ties: bool = False
    random_state: Optional[int] = None
    regression_loss: RegressionLoss = RegressionLoss.epsilon_insensitive
    epsilon: float = 0.0


def build_config(
    svm_type: SVMType,
    C: Optional[float],
    nu: Optional[float],
    penalty: Optional[Penalty],
    loss: Optional[Loss],
    dual: Optional[Union[str, bool]],
    multi_class: Optional[MultiClass],
    fit_intercept: Optional[bool],
    intercept_scalling: Optional[float],
    kernel: Optional[Kernel],
    degree: Optional[int],
    gamma: Optional[Union[float, str]],
    coef0: Optional[float],
    shrinking: Optional[bool],
    probability: Optional[bool],
    tol: Optional[float],
    cache_size: Optional[float],
    class_weight: Optional[Union[dict, str]],
    verbose: Optional[Union[bool, int]],
    max_iter: Optional[int],
    decision_function_shape: Optional[DecisionFunctionShape],
    break_ties: Optional[bool],
    random_state: Optional[int],
    regression_loss: Optional[RegressionLoss],
    epsilon: Optional[float],
) -> Config:
    config = Config

    if C is not None:
        if svm_type == SVMType.Nu:
            logger.warning(f"C specified but NuSVM is chosen. Ignoring C")
        elif C <= 0.0:
            logger.error(f"Value of C must be strictly positive. Input Value: {C}")
            raise ValueError("Non-positive C")
        else:
            config.C = C
    if nu is not None:
        if svm_type != SVMType.Nu:
            logger.warning(f"nu specified but NuSVM is not chosen, Ignoring nu")
        elif nu <= 0.0 or nu > 1.0:
            logger.error(f"Value of nu must be in (0, 1], Input Value: {nu}.")
            raise ValueError("Out of bound Nu")
        else:
            config.nu = nu

    if svm_type == SVMType.Linear:
        if penalty is not None:
            config.penalty = penalty
        if epsilon is not None:
            config.epsilon = epsilon
        if loss is not None:
            config.loss = loss
        if regression_loss is not None:
            config.regression_loss = regression_loss
        if dual is not None:
            if dual == "auto" or isinstance(dual, bool):
                config.dual = dual
            else:
                logger.error(f"dual must be auto, True or False. Input Value: {dual}")
                raise ValueError("Unsupported dual")
        if multi_class is not None:
            config.multi_class = multi_class
        if fit_intercept is not None:
            config.fit_intercept = fit_intercept
        if intercept_scalling is not None:
            config.intercept_scalling = intercept_scalling
    else:
        if kernel is not None:
            config.kernel = kernel
        if degree is not None:
            if kernel != Kernel.poly:
                logger.warning(
                    f"degree parameter specified but kernel is not polynomial. Ingoring degree value specified."
                )
                logger.warning(
                    f"degree only significant for poly kernel type, current kernel type: {config.kernel}."
                )
            elif degree < 0:
                logger.error(f"Degree must be non-negative, Input Value: {degree}")
                raise ValueError("Negative degree")
            else:
                config.degree = degree
        if gamma is not None:
            if gamma == "scale" or gamma == "auto":
                pass
            elif not isinstance(gamma, float):
                logger.error(
                    f"Unknown gamma, must be 'scale', 'auto' or floating number. Input Value: {gamma}"
                )
                raise ValueError("Unsupported gamma value")
            elif gamma < 0:
                logger.error(f"gamma must be non-negative, Input Value: {gamma}")
                raise ValueError("Negative gamma")
            config.gamma = gamma
        if coef0 is not None:
            if kernel != Kernel.poly and kernel != Kernel.sigmoid:
                logger.warning(
                    "coef0 specified but kernel is neither polynomial nor sigmoid."
                )
                logger.warning(
                    "Ignoring coef0, it is only significant for polynomial or sigmoid kernel"
                )
            else:
                config.coef0 = coef0
        if shrinking is not None:
            config.shrinking = shrinking
        if probability is not None:
            config.probability = probability
        if cache_size is not None:
            config.cache_size = cache_size
        if break_ties is not None:
            config.break_ties = break_ties
        if decision_function_shape is not None:
            config.decision_function_shape = decision_function_shape

    if tol is not None:
        config.tol = tol
    elif svm_type == SVMType.Linear:
        config.tol = 1e-4

    if class_weight is not None:
        if class_weight == "balanced" or isinstance(class_weight, dict):
            config.class_weight = class_weight
    if verbose is not None:
        if svm_type == SVMType.Linear and not isinstance(verbose, int):
            logger.error(
                f"verbose must be integer for LinearSVC. Input Value: {verbose}"
            )
            raise ValueError("Non integer verbose")
        if svm_type != SVMType.Linear and not isinstance(verbose, bool):
            logger.error(
                f"verbose must be bool for SVC or NuSVC. Input Value: {verbose}"
            )
            raise ValueError("None bool verbose")
        config.verbose = verbose
    else:
        config.verbose = 0
    if max_iter is not None:
        config.max_iter = max_iter
    elif svm_type == SVMType.Linear:
        config.max_iter = 1000
    if random_state is not None:
        if config.probability is False and svm_type != SVMType.Linear:
            logger.warning("random state specified but probability is False.")
            logger.warning("To make use of random state, enable probability.")
        elif config.dual is False and svm_type == SVMType.Linear:
            logger.warning("random state specified but dual is False.")
            logger.warning("To make use of random state, enable dual.")
        else:
            config.random_state = random_state

    return config


def coerce_value(field_type, value):
    """Coerce CLI input (string) into correct type."""
    if value is None:
        return None
    origin_type = getattr(field_type, "__origin__", None)

    # Handle Optional or Union
    if origin_type is Union:
        for arg in field_type.__args__:
            if arg is type(None):
                continue
            try:
                return coerce_value(arg, value)
            except Exception:
                continue
        raise typer.BadParameter(f"Could not parse value: {value}")

    # Handle Enum
    if isinstance(field_type, type) and issubclass(field_type, Enum):
        return field_type(value)

    # Handle bool
    if field_type is bool:
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)

    # Handle dict
    if field_type is dict or field_type == dict:
        return ast.literal_eval(value)

    # Handle tuple
    if origin_type is tuple or field_type is tuple:
        try:
            parsed = ast.literal_eval(value)
            if not isinstance(parsed, tuple):
                raise ValueError(f"Expected tuple, got {type(parsed)}")

            # If specific tuple element types are specified, coerce each
            args = get_args(field_type)
            if args:
                if len(args) == 2 and args[1] is Ellipsis:
                    # Tuple[int, ...]
                    return tuple(coerce_value(args[0], v) for v in parsed)
                elif len(args) == len(parsed):
                    return tuple(coerce_value(t, v) for t, v in zip(args, parsed))
                else:
                    raise typer.BadParameter(
                        f"Expected {len(args)} elements, got {len(parsed)}"
                    )
            return parsed
        except Exception as e:
            raise typer.BadParameter(f"Invalid tuple input: {value!r}. Error: {e}")

    # Default: cast to field_type
    return field_type(value)


def classification_logic(
    train_data_path: str,
    test_data_path: str,
    output_result_path: str,
    svm_type: SVMType,
    label_name: Optional[str] = "label",
    preview_prediction_result: Optional[bool] = False,
    do_explain_model: Optional[bool] = False,
    shap_output_path: Optional[str] = "./shap_values_output.csv",
    C: Optional[float] = None,
    nu: Optional[float] = None,
    penalty: Optional[Penalty] = None,
    loss: Optional[Loss] = None,
    dual: Optional[Union[str, bool]] = None,
    multi_class: Optional[MultiClass] = None,
    fit_intercept: Optional[bool] = None,
    intercept_scalling: Optional[float] = None,
    kernel: Optional[Kernel] = None,
    degree: Optional[int] = None,
    gamma: Optional[Union[float, str]] = None,
    coef0: Optional[float] = None,
    shrinking: Optional[bool] = None,
    probability: Optional[bool] = None,
    tol: Optional[float] = None,
    cache_size: Optional[float] = None,
    class_weight: Optional[Union[dict, str]] = None,
    verbose: Optional[Union[bool, int]] = None,
    max_iter: Optional[int] = None,
    decision_function_shape: Optional[DecisionFunctionShape] = None,
    break_ties: Optional[bool] = None,
    random_state: Optional[int] = None,
):
    config = build_config(
        svm_type=svm_type,
        C=C,
        nu=nu,
        penalty=penalty,
        loss=loss,
        dual=dual,
        multi_class=multi_class,
        fit_intercept=fit_intercept,
        intercept_scalling=intercept_scalling,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        shrinking=shrinking,
        probability=probability,
        tol=tol,
        cache_size=cache_size,
        class_weight=class_weight,
        verbose=verbose,
        max_iter=max_iter,
        decision_function_shape=decision_function_shape,
        break_ties=break_ties,
        random_state=random_state,
        regression_loss=None,
        epsilon=None,
    )

    if svm_type == SVMType.C:
        model = svm.SVC(
            C=config.C,
            kernel=config.kernel,
            degree=config.degree,
            gamma=config.gamma,
            coef0=config.coef0,
            shrinking=config.shrinking,
            probability=config.probability,
            tol=config.tol,
            cache_size=config.cache_size,
            class_weight=config.class_weight,
            verbose=config.verbose,
            max_iter=config.max_iter,
            decision_function_shape=config.decision_function_shape,
            break_ties=config.break_ties,
            random_state=config.random_state,
        )
    elif svm_type == SVMType.Nu:
        model = svm.NuSVC(
            nu=config.nu,
            kernel=config.kernel,
            degree=config.degree,
            gamma=config.gamma,
            coef0=config.coef0,
            shrinking=config.shrinking,
            probability=config.probability,
            tol=config.tol,
            cache_size=config.cache_size,
            class_weight=config.class_weight,
            verbose=config.verbose,
            max_iter=config.max_iter,
            decision_function_shape=config.decision_function_shape,
            break_ties=config.break_ties,
            random_state=config.random_state,
        )
    elif svm_type == SVMType.Linear:
        model = svm.LinearSVC(
            penalty=config.penalty,
            loss=config.loss,
            dual=config.dual,
            tol=config.tol,
            C=config.C,
            multi_class=config.multi_class,
            fit_intercept=config.fit_intercept,
            intercept_scaling=config.intercept_scaling,
            class_weight=config.class_weight,
            verbose=config.verbose,
            random_state=config.random_state,
            max_iter=config.max_iter,
        )
    else:
        logger.error(f"Unknown SVM Type, Supported types: {SVMType}")
        raise Exception("Unknown SVM Type")

    logger.info("Loading training data.")
    train_X, train_y = load_data(train_data_path, label_name)
    logger.info("Training model.")
    model.fit(train_X, train_y)
    logger.info("Finished model training.")

    logger.info("Loading testing data.")
    test_X = load_data(test_data_path)
    test_X = test_X[train_X.columns]
    logger.info("Predicting test data")
    predictions = model.predict(test_X)
    test_pred = test_X.copy()
    test_pred["prediction"] = predictions
    test_pred.to_csv(output_result_path, index=False)
    logger.info(f"Prediction Finished, result saved to {output_result_path}")

    if preview_prediction_result is True:
        print(test_pred)

    if do_explain_model is True:
        if svm_type == SVMType.Linear or kernel == Kernel.linear:
            explainer_type = "linear"
        else:
            explainer_type = "kernel"
        explain_model(
            model,
            train_X,
            test_X,
            train_y,
            predictions,
            explainer_type,
            shap_output_path,
        )


# typer.option default set to None, Actual default value already set when initialize Config object.
@svm_app.command()
def classification(
    svm_type: SVMType = "C",
    train_data_path: str = "example/data/classification_train.csv",
    test_data_path: str = "example/data/classification_test.csv",
    output_result_path: str = "example/data/classification_output.csv",
    shap_output_path: Optional[str] = typer.Option(
        None, help="Output folder for SHAP plot and values."
    ),
    preview_prediction_result: bool = False,
    label_name: str = "label",
    do_explain_model: bool = False,
    C: Optional[float] = typer.Option(
        None,
        help="Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. Default: 1.0",
        show_default=False,
    ),
    nu: Optional[float] = typer.Option(
        None,
        help="Upper bound on the fraction of margin errors and a lower bound of the fraction of support vectors. Must be in (0, 1]. Default: 0.5",
        show_default=False,
    ),
    penalty: Optional[Penalty] = typer.Option(
        Penalty.l2,
        help="Norm used in the penalization. Default: l2",
        show_default=False,
    ),
    loss: Optional[Loss] = typer.Option(
        Loss.squared_hinge,
        help="Loss function. Default: squared_hinge",
        show_default=False,
    ),
    dual: Optional[str] = typer.Option(
        None,
        help="Select the algorithm to either solve the dual or primal optimization problem. Avaiable option: true, false, auto. Default: auto",
        show_default=False,
    ),
    multi_class: Optional[MultiClass] = typer.Option(
        None,
        help="Determines the multi-class strategy if y contains ore than two classes. Default: 'ovr'",
        show_default=False,
    ),
    fit_intercept: Optional[bool] = typer.Option(
        None,
        help="Whether or not to fit an intercept. Default: True",
        show_default=False,
    ),
    intercept_scalling: Optional[float] = typer.Option(
        None,
        help="Allow intercept term to have a different regularization behavior compared to the other features, Default: 1.0",
        show_default=False,
    ),
    kernel: Optional[Kernel] = typer.Option(
        Kernel.rbf, help="Kernal Type. Default: rbf", show_default=False
    ),
    degree: Optional[int] = typer.Option(
        None,
        help="Degree of the polynomial kernel function. Default: 3",
        show_default=False,
    ),
    gamma: Optional[str] = typer.Option(
        None,
        help="Kernel coefficient for 'rbf', 'poly' and sigmoid'. Default: 'scale'",
        show_default=False,
    ),
    coef0: Optional[float] = typer.Option(
        None,
        help="Independent term in kernel function. Default: 0.0",
        show_default=False,
    ),
    shrinking: Optional[bool] = typer.Option(
        None,
        help="Whether to use the shrinking heuristic, Default: True",
        show_default=False,
    ),
    probability: Optional[bool] = typer.Option(
        None,
        help="Whether to enable probability estimates. Default: False",
        show_default=False,
    ),
    tol: Optional[float] = typer.Option(
        None,
        help="Tolerance for stopping criterion. Default: [Linear] 1e-4, [C|Nu] 1e-3",
        show_default=False,
    ),
    cache_size: Optional[float] = typer.Option(
        None, help="Size of the kernel cache (in MB). Default: 200", show_default=False
    ),
    class_weight: Optional[str] = typer.Option(
        None,
        help="Parameter of [C|Nu|Linear] of class i to class_weight[i] * [C|Nu|Linear]. Default: All weight are one",
        show_default=False,
    ),
    verbose: Optional[str] = typer.Option(
        None, help="Enable verbose output. Default: False", show_default=False
    ),
    max_iter: Optional[int] = typer.Option(
        None,
        help="Hard limit on iterations within solver, or -1 for no limit. Default: [C|Nu] -1, [Linear]: 1000",
        show_default=False,
    ),
    decision_function_shape: Optional[DecisionFunctionShape] = typer.Option(
        DecisionFunctionShape.ovr,
        help="Whether to return a one-vs-rest (ovr) or one-vs-one (ovo) decision function. Default: ovr",
        show_default=False,
    ),
    break_ties: Optional[bool] = typer.Option(
        None,
        help="Whether predict will break ties according to the confidence values of decision_function. Default=False",
        show_default=False,
    ),
    random_state: Optional[int] = typer.Option(
        None,
        help="Controls the pseudo random number generation. Default=None",
        show_default=False,
    ),
):

    dual = coerce_value(Config.__annotations__["dual"], dual)
    gamma = coerce_value(Config.__annotations__["gamma"], gamma)
    class_weight = coerce_value(Config.__annotations__["class_weight"], class_weight)
    verbose = coerce_value(Config.__annotations__["verbose"], verbose)

    classification_logic(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        output_result_path=output_result_path,
        shap_output_path=shap_output_path,
        preview_prediction_result=preview_prediction_result,
        label_name=label_name,
        do_explain_model=do_explain_model,
        svm_type=svm_type,
        C=C,
        nu=nu,
        penalty=penalty,
        loss=loss,
        dual=dual,
        multi_class=multi_class,
        fit_intercept=fit_intercept,
        intercept_scalling=intercept_scalling,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        shrinking=shrinking,
        probability=probability,
        tol=tol,
        cache_size=cache_size,
        class_weight=class_weight,
        verbose=verbose,
        max_iter=max_iter,
        decision_function_shape=decision_function_shape,
        break_ties=break_ties,
        random_state=random_state,
    )


def regression_logic(
    train_data_path: str,
    test_data_path: str,
    output_result_path: str,
    svm_type: SVMType,
    shap_output_path: Optional[str] = "./shap_values_output.csv",
    preview_prediction_result: Optional[bool] = False,
    label_name: Optional[str] = "label",
    do_explain_model: Optional[bool] = False,
    C: Optional[float] = None,
    nu: Optional[float] = None,
    loss: Optional[RegressionLoss] = None,
    dual: Optional[Union[str, bool]] = None,
    fit_intercept: Optional[bool] = None,
    intercept_scalling: Optional[float] = None,
    kernel: Optional[Kernel] = None,
    degree: Optional[int] = None,
    gamma: Optional[Union[float, str]] = None,
    coef0: Optional[float] = None,
    shrinking: Optional[bool] = None,
    tol: Optional[float] = None,
    cache_size: Optional[float] = None,
    verbose: Optional[Union[bool, int]] = None,
    max_iter: Optional[int] = None,
    epsilon: Optional[float] = None,
):

    config = build_config(
        svm_type=svm_type,
        C=C,
        nu=nu,
        penalty=None,
        loss=None,
        regression_loss=loss,
        dual=dual,
        multi_class=None,
        fit_intercept=fit_intercept,
        intercept_scalling=intercept_scalling,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        shrinking=shrinking,
        probability=None,
        tol=tol,
        cache_size=cache_size,
        class_weight=None,
        verbose=verbose,
        max_iter=max_iter,
        decision_function_shape=None,
        break_ties=None,
        random_state=None,
        epsilon=epsilon,
    )

    if svm_type == SVMType.C:
        model = svm.SVR(
            C=config.C,
            kernel=config.kernel,
            degree=config.degree,
            gamma=config.gamma,
            coef0=config.coef0,
            shrinking=config.shrinking,
            tol=config.tol,
            cache_size=config.cache_size,
            verbose=config.verbose,
            max_iter=config.max_iter,
        )
    elif svm_type == SVMType.Nu:
        model = svm.NuSVR(
            nu=config.nu,
            kernel=config.kernel,
            degree=config.degree,
            gamma=config.gamma,
            coef0=config.coef0,
            shrinking=config.shrinking,
            tol=config.tol,
            cache_size=config.cache_size,
            verbose=config.verbose,
            max_iter=config.max_iter,
        )
    elif svm_type == SVMType.Linear:
        model = svm.LinearSVR(
            loss=config.regression_loss,
            epsilon=config.epsilon,
            dual=config.dual,
            tol=config.tol,
            C=config.C,
            fit_intercept=config.fit_intercept,
            intercept_scaling=config.intercept_scaling,
            verbose=config.verbose,
            random_state=config.random_state,
            max_iter=config.max_iter,
        )
    else:
        logger.error(f"Unknown SVM Type, Supported types: {SVMType}")
        raise Exception("Unknown SVM Type")

    logger.info("Loading training data.")
    train_X, train_y = load_data(train_data_path, label_name)
    logger.info("Training model.")
    model.fit(train_X, train_y)
    logger.info("Finished model training.")

    logger.info("Loading testing data.")
    test_X = load_data(test_data_path)
    test_X = test_X[train_X.columns]
    logger.info("Predicting test data")
    predictions = model.predict(test_X)
    test_pred = test_X.copy()
    test_pred["prediction"] = predictions
    test_pred.to_csv(output_result_path, index=False)
    logger.info(f"Prediction Finished, result saved to {output_result_path}")

    if preview_prediction_result is True:
        print(test_pred)

    if do_explain_model is True:
        if svm_type == SVMType.Linear or kernel == Kernel.linear:
            explainer_type = "linear"
        else:
            explainer_type = "kernel"
        explain_model(
            model,
            train_X,
            test_X,
            train_y,
            predictions,
            explainer_type,
            shap_output_path,
        )


# typer.option default set to None, Actual default value already set when initialize Config object.
@svm_app.command()
def regression(
    svm_type: SVMType = "C",
    train_data_path: str = "example/data/regression_train.csv",
    test_data_path: str = "example/data/regression_test.csv",
    output_result_path: str = "example/data/regression_output.csv",
    shap_output_path: Optional[str] = typer.Option(
        None, help="Output folder for SHAP plot and values."
    ),
    preview_prediction_result: bool = False,
    label_name: str = "label",
    do_explain_model: bool = False,
    C: Optional[float] = typer.Option(
        None,
        help="Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. Default: 1.0",
        show_default=False,
    ),
    nu: Optional[float] = typer.Option(
        None,
        help="Upper bound on the fraction of margin errors and a lower bound of the fraction of support vectors. Must be in (0, 1]. Default: 0.5",
        show_default=False,
    ),
    epsilon: Optional[float] = typer.Option(
        None,
        help="Epsilon parameter in the epsilon-insensitive loss function. Default: 0.0",
        show_default=False,
    ),
    loss: Optional[RegressionLoss] = typer.Option(
        RegressionLoss.epsilon_insensitive,
        help="Loss function. Default: epsilon_insensitive",
        show_default=False,
    ),
    dual: Optional[str] = typer.Option(
        None,
        help="Select the algorithm to either solve the dual or primal optimization problem. Avaiable option: true, false, auto. Default: auto",
        show_default=False,
    ),
    fit_intercept: Optional[bool] = typer.Option(
        None,
        help="Whether or not to fit an intercept. Default: True",
        show_default=False,
    ),
    intercept_scalling: Optional[float] = typer.Option(
        None,
        help="Allow intercept term to have a different regularization behavior compared to the other features, Default: 1.0",
        show_default=False,
    ),
    kernel: Optional[Kernel] = typer.Option(
        Kernel.rbf, help="Kernal Type. Default: rbf", show_default=False
    ),
    degree: Optional[int] = typer.Option(
        None,
        help="Degree of the polynomial kernel function. Default: 3",
        show_default=False,
    ),
    gamma: Optional[str] = typer.Option(
        None,
        help="Kernel coefficient for 'rbf', 'poly' and sigmoid'. Default: 'scale'",
        show_default=False,
    ),
    coef0: Optional[float] = typer.Option(
        None,
        help="Independent term in kernel function. Default: 0.0",
        show_default=False,
    ),
    shrinking: Optional[bool] = typer.Option(
        None,
        help="Whether to use the shrinking heuristic, Default: True",
        show_default=False,
    ),
    tol: Optional[float] = typer.Option(
        None,
        help="Tolerance for stopping criterion. Default: [Linear] 1e-4, [C|Nu] 1e-3",
        show_default=False,
    ),
    cache_size: Optional[float] = typer.Option(
        None, help="Size of the kernel cache (in MB). Default: 200", show_default=False
    ),
    verbose: Optional[str] = typer.Option(
        None, help="Enable verbose output. Default: False", show_default=False
    ),
    max_iter: Optional[int] = typer.Option(
        None,
        help="Hard limit on iterations within solver, or -1 for no limit. Default: [C|Nu] -1, [Linear]: 1000",
        show_default=False,
    ),
):

    dual = coerce_value(Config.__annotations__["dual"], dual)
    gamma = coerce_value(Config.__annotations__["gamma"], gamma)
    verbose = coerce_value(Config.__annotations__["verbose"], verbose)

    regression_logic(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        output_result_path=output_result_path,
        shap_output_path=shap_output_path,
        preview_prediction_result=preview_prediction_result,
        label_name=label_name,
        do_explain_model=do_explain_model,
        svm_type=svm_type,
        C=C,
        nu=nu,
        loss=loss,
        dual=dual,
        fit_intercept=fit_intercept,
        intercept_scalling=intercept_scalling,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        shrinking=shrinking,
        tol=tol,
        cache_size=cache_size,
        verbose=verbose,
        max_iter=max_iter,
        epsilon=epsilon,
    )
