from enum import Enum
from typing import Optional, Union
from dataclasses import dataclass

from click import Option
from numpy import isin
from sklearn.utils import class_weight
import typer
from sklearn import svm

from .utils import load_data
import ast
import logging

model_app = typer.Typer()
logger = logging.getLogger(__name__)


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
    l1 = 'l1'
    l2 = 'l2'

class Loss(str, Enum):
    hinge = 'hinge'
    squared_hinge = 'squared_hinge'

class MultiClass(str, Enum):
    ovr = 'ovr'
    crammer_singer = 'crammer_singer'

@dataclass
class Config:
    C: float = 1.0
    nu: float = 0.5
    penalty: Penalty = Penalty.l2
    loss: Loss = Loss.squared_hinge
    dual: Union[str, bool] = 'auto'
    multi_class: MultiClass = MultiClass.ovr
    fit_intercept: bool = True
    intercept_scalling: float = 1.0
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
    random_state: Optional[int]
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
        if loss is not None:
            config.loss = loss
        if dual is not  None:
            if dual == 'auto' or isinstance(dual, bool):
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
                logger.warning(f"degree parameter specified but kernel is not polynomial. Ingoring degree value specified.")
                logger.warning(f"degree only significant for poly kernel type, current kernel type: {config.kernel}.")
            elif degree < 0:
                logger.error(f"Degree must be non-negative, Input Value: {degree}")
                raise ValueError("Negative degree")
            else:
                config.degree = degree
        if gamma is not None:
            if gamma == "scale" or gamma == "auto":
                pass
            elif not isinstance(gamma, float):
                logger.error(f"Unknown gamma, must be 'scale', 'auto' or floating number. Input Value: {gamma}")
                raise ValueError("Unsupported gamma value")
            elif gamma < 0:
                logger.error(f"gamma must be non-negative, Input Value: {gamma}")
                raise ValueError("Negative gamma")
            config.gamma = gamma
        if coef0 is not None:
            if kernel != Kernel.poly and kernel != Kernel.sigmoid:
                logger.warning("coef0 specified but kernel is neither polynomial nor sigmoid.")
                logger.warning("Ignoring coef0, it is only significant for polynomial or sigmoid kernel")
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
            logger.error(f"verbose must be integer for LinearSVC. Input Value: {verbose}")
            raise ValueError("Non integer verbose")
        if svm_type != SVMType.Linear and not isinstance(verbose, bool):
            logger.error(f"verbose must be bool for SVC or NuSVC. Input Value: {verbose}")
            raise ValueError("None bool verbose")
        config.verbose = verbose
    else:
        config.verbose = 0
    if max_iter is not None:
        config.max_iter = max_iter
    elif svm_type == SVMType.Linear:
        config.max_iter = 1000
    if random_state is not None:
        if config.probability == False and svm_type != SVMType.Linear:
            logger.warning("random state specified but probability is False.")
            logger.warning("To make use of random state, enable probability.")
        elif config.dual == False and svm_type == SVMType.Linear:
            logger.warning("random state specified but dual is False.")
            logger.warning("To make use of random state, enable dual.")
        else:
            config.random_state = random_state

    return config

def coerce_value(field_type, value):
    """Coerce CLI input (string) into correct type."""
    if value is None:
        return None
    origin_type = getattr(field_type, '__origin__', None)

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

    # Default: cast to field_type
    return field_type(value)



@model_app.command()
def classification(
    svm_type: SVMType = "C",
    train_data_path: str = "example/data/classification_train.csv",
    test_data_path: str = "example/data/classification_test.csv",
    label_name: str = "label",
    C: Optional[float] = None,
    nu: Optional[float] = None,
    penalty: Optional[Penalty] = None,
    loss: Optional[Loss] = None,
    dual: Optional[str] = None,
    multi_class: Optional[MultiClass] = None,
    fit_intercept: Optional[bool] = None,
    intercept_scalling: Optional[float] = None,
    kernel: Optional[Kernel] = None,
    degree: Optional[int] = None,
    gamma: Optional[str] = None,
    coef0: Optional[float] = None,
    shrinking: Optional[bool] = None,
    probability: Optional[bool] = None,
    tol: Optional[float] = None,
    cache_size: Optional[float] = None,
    class_weight: Optional[str] = None,
    verbose: Optional[str] = None,
    max_iter: Optional[int] = None,
    decision_function_shape: Optional[DecisionFunctionShape] = None,
    break_ties: Optional[bool] = None,
    random_state: Optional[int] = None,
):

    dual = coerce_value(Config.__annotations__['dual'], dual)
    gamma = coerce_value(Config.__annotations__['gamma'], gamma)
    class_weight = coerce_value(Config.__annotations__['class_weight'], class_weight)
    verbose = coerce_value(Config.__annotations__['verbose'], verbose)

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
            random_state=random_state)

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
            random_state=config.random_state
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
            random_state=config.random_state
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
            max_iter=config.max_iter
        )
    else:
        logger.error(f"Unknown SVM Type, Supported types: {SVMTYPE}")
        raise Exception("Unknown SVM Type")

    train_X, train_y = load_data(train_data_path, label_name)
    model.fit(train_X, train_y)

    test_X = load_data(test_data_path)
    test_pred = model.predict(test_X)
    print(test_pred)
