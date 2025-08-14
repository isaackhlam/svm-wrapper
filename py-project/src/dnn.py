import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import typer
from sklearn import neural_network

from .svm import coerce_value
from .utils import load_data, explain_model

dnn_app = typer.Typer()
logger = logging.getLogger(__name__)


class Loss(str, Enum):
    squared_error = "squared_error"
    poisson = "poisson"


class Activation(str, Enum):
    identity = "identity"
    logistic = "logistic"
    tanh = "tanh"
    relu = "relu"


class Solver(str, Enum):
    lbfgs = "lbfgs"
    sgd = "sgd"
    adam = "adam"


class LearningRate(str, Enum):
    constant = "constant"
    invscaling = "invscaling"
    adaptive = "adaptive"


@dataclass
class Config:
    loss: Loss = Loss.squared_error
    hidden_layer_sizes: tuple = (100,)
    activation: Activation = Activation.relu
    solver: Solver = Solver.adam
    alpha: float = 0.0001
    batch_size: Union[str, float] = "auto"  # only accept str is auto
    learning_rate: LearningRate = LearningRate.constant
    learning_rate_init: float = 0.001
    power_t: float = 0.5
    max_iter: int = 200
    shuffle: bool = True
    random_state: Optional[int] = None
    tol: float = 1e-4
    verbose: bool = False
    warm_start: bool = False
    momentum: float = 0.9
    nesterovs_momentum: bool = True
    early_stopping: bool = False
    validation_fraction: float = 0.1
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    n_iter_no_change: int = 10
    max_fun: int = 15000


def build_config(
    loss: Optional[Loss],
    hidden_layer_sizes: Optional[tuple],
    activation: Optional[Activation],
    solver: Optional[Solver],
    alpha: Optional[float],
    batch_size: Optional[Union[str, float]],
    learning_rate: Optional[LearningRate],
    learning_rate_init: Optional[float],
    power_t: Optional[float],
    max_iter: Optional[int],
    shuffle: Optional[bool],
    random_state: Optional[int],
    tol: Optional[float],
    verbose: Optional[bool],
    warm_start: Optional[bool],
    momentum: Optional[float],
    nesterovs_momentum: Optional[bool],
    early_stopping: Optional[bool],
    validation_fraction: Optional[float],
    beta_1: Optional[float],
    beta_2: Optional[float],
    epsilon: Optional[float],
    n_iter_no_change: Optional[int],
    max_fun: Optional[int],
) -> Config:
    config = Config

    if loss is not None:
        config.loss = loss
    if hidden_layer_sizes is not None:
        config.hidden_layer_sizes = hidden_layer_sizes
    if activation is not None:
        config.activation = activation
    if solver is not None:
        config.solver = solver
    if alpha is not None:
        config.alpha = alpha
    if batch_size is not None:
        if batch_size == "auto" or isinstance(batch_size, int):
            config.batch_size == batch_size
        else:
            logging.error(
                f"batch_size must be 'auto' or int. Input Value: {batch_size}"
            )
            raise ValueError("Unsupported batch_size")
    if learning_rate is not None:
        config.learning_rate = learning_rate
    if learning_rate_init is not None:
        if config.solver != Solver.sgd and config.solver != Solver.adam:
            logger.warning(
                "learning_rate_init specified but solver is not 'sgd' or 'adam'. Ignoring learning_rate_init."
            )
        else:
            config.learning_rate_init = learning_rate_init
    if power_t is not None:
        if config.solver != Solver.sgd:
            logger.warning(
                "power_t specified but solver is not 'sgd'. Ignoring power_t."
            )
        else:
            config.power_t = power_t
    if max_iter is not None:
        config.max_iter = max_iter
    if shuffle is not None:
        config.shuffle = shuffle
    if random_state is not None:
        config.random_state = random_state
    if tol is not None:
        config.tol = tol
    if verbose is not None:
        config.verbose = verbose
    if warm_start is not None:
        config.warm_start = warm_start
    if momentum is not None:
        config.momentum = momentum
    if nesterovs_momentum is not None:
        if config.solver == Solver.sgd and config.momentum > 0:
            config.nesterovs_momentum = nesterovs_momentum
        else:
            logger.warning(
                "nesterovs_momentum specified but solver is not sgd, or momentum less than 0. Ignoring nesterovs_momentum"
            )
    if early_stopping is not None:
        config.early_stopping = early_stopping
    if validation_fraction is not None:
        if config.early_stopping is True:
            config.validation_fraction = validation_fraction
        else:
            logger.warning(
                "validation_fraction specified but early_stopping is False. Ignoring validation_fraction"
            )
    if beta_1 is not None:
        if config.solver == Solver.adam:
            config.beta_1 = beta_1
        else:
            logger.warning("beta_1 specified but solver is not 'adam'. Ignoring beta_1")
    if beta_2 is not None:
        if config.solver == Solver.adam:
            config.beta_2 = beta_2
        else:
            logger.warning("beta_2 specified but solver is not 'adam'. Ignoring beta_2")
    if epsilon is not None:
        if config.solver == Solver.adam:
            config.epsilon = epsilon
        else:
            logger.warning(
                "epsilon specified but solver is not 'adam'. Ignoring epsilon"
            )
    if n_iter_no_change is not None:
        if config.solver != Solver.sgd and config.solver != Solver.adam:
            logger.warning(
                "n_iter_no_change specified but solver is not 'sgd' or 'adam'. Ignoring n_iter_no_change."
            )
        else:
            config.n_iter_no_change = n_iter_no_change
    if max_fun is not None:
        if config.solver == Solver.lbfgs:
            config.max_fun = max_fun
        else:
            logger.warning(
                "max_fun specified but solver is not 'lbfgs'. Ignoring max_fun"
            )

    return config


@dnn_app.command()
def classification(
    train_data_path: str = "example/data/classification_train.csv",
    test_data_path: str = "example/data/classification_test.csv",
    output_result_path: str = "example/data/classification_dnn_output.csv",
    shap_output_path: Optional[str] = typer.Option(
        None, help="Output folder for SHAP plot and values."
    ),
    preview_prediction_result: bool = False,
    label_name: str = "label",
    do_explain_model: bool = False,
    hidden_layer_sizes: Optional[str] = typer.Option(
        None,
        help="The ith element represents the number of neurons in the ith hidden layer. Default: (100,)",
        show_default=False,
    ),
    activation: Optional[Activation] = typer.Option(
        None,
        help="Activation function for the hidden layer. Default: relu",
        show_default=False,
    ),
    solver: Optional[Solver] = typer.Option(
        None,
        help="The solver for weight optimization. Default 'adam'",
        show_default=False,
    ),
    alpha: Optional[float] = typer.Option(
        None,
        help="Strength of the L2 regularization term. Default: 0.0001",
        show_default=False,
    ),
    batch_size: Optional[str] = typer.Option(
        None,
        help="Size of minibatches for stochastic optimizer. Default: 'auto'",
        show_default=False,
    ),
    learning_rate: Optional[LearningRate] = typer.Option(
        None,
        help="Learning rate schedule for weight updates Default: 'constant'",
        show_default=False,
    ),
    learning_rate_init: Optional[float] = typer.Option(
        None, help="The initial learning rate used. Default: 0.001", show_default=False
    ),
    power_t: Optional[float] = typer.Option(
        None,
        help="The exponent for inverse scaling learning rate. Default: 0.5",
        show_default=False,
    ),
    max_iter: Optional[int] = typer.Option(
        None, help="Maximum number of iterations. Default: 200", show_default=False
    ),
    shuffle: Optional[bool] = typer.Option(
        None,
        help="Whether to shuffle samples in each iteration. Default: True",
        show_default=True,
    ),
    random_state: Optional[int] = typer.Option(
        None,
        help="Determines random number generation. Default: None",
        show_default=False,
    ),
    tol: Optional[float] = typer.Option(
        None, help="Tolerance for the optimization. Default: 1e-4", show_default=False
    ),
    verbose: Optional[bool] = typer.Option(
        None,
        help="Whether to print progress messags to stdout. Default: False",
        show_default=False,
    ),
    warm_start: Optional[bool] = typer.Option(
        None,
        help="Whether reuse the solution of the previous call to fit as initialization. Default: False",
        show_default=False,
    ),
    momentum: Optional[float] = typer.Option(
        None,
        help="Momentum for gradient descent update. Default: 0.9",
        show_default=False,
    ),
    nesterovs_momentum: Optional[bool] = typer.Option(
        None,
        help="Whether to use Nesterov's Momentum. Default: True",
        show_default=False,
    ),
    early_stopping: Optional[bool] = typer.Option(
        None,
        help="Whether to use early stopping to terminate training when validation score is not improving. Default: False",
        show_default=False,
    ),
    validation_fraction: Optional[float] = typer.Option(
        None,
        help="The proportion of training data to set aside as validation set for early stopping. Default: 0.1",
        show_default=False,
    ),
    beta_1: Optional[float] = typer.Option(
        None,
        help="Exponential decay rate for estimates of first moment vector in adam. Default: 0.9",
        show_default=False,
    ),
    beta_2: Optional[float] = typer.Option(
        None,
        help="Exponential decay rate for estimates of second moment vector in adam. Default: 0.999",
        show_default=False,
    ),
    epsilon: Optional[float] = typer.Option(
        None,
        help="Value for numerical stability in adam. Default: 1e-8",
        show_default=False,
    ),
    n_iter_no_change: Optional[int] = typer.Option(
        None,
        help="Maximum number of epochs to not meet tol improvement. Default: 10",
        show_default=False,
    ),
    max_fun: Optional[int] = typer.Option(
        None,
        help="Maximum number of function calls. Only used when solver='lbfgs'. Default: 15000",
        show_default=False,
    ),
):

    hidden_layer_sizes = coerce_value(
        Config.__annotations__["hidden_layer_sizes"], hidden_layer_sizes
    )
    batch_size = coerce_value(Config.__annotations__["batch_size"], batch_size)

    config = build_config(
        loss=None,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        power_t=power_t,
        max_iter=max_iter,
        shuffle=shuffle,
        random_state=random_state,
        tol=tol,
        verbose=verbose,
        warm_start=warm_start,
        momentum=momentum,
        nesterovs_momentum=nesterovs_momentum,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        n_iter_no_change=n_iter_no_change,
        max_fun=max_fun,
    )
    model = neural_network.MLPClassifier(
        hidden_layer_sizes=config.hidden_layer_sizes,
        activation=config.activation,
        solver=config.solver,
        alpha=config.alpha,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        learning_rate_init=config.learning_rate_init,
        power_t=config.power_t,
        max_iter=config.max_iter,
        shuffle=config.shuffle,
        random_state=config.random_state,
        tol=config.tol,
        verbose=config.verbose,
        warm_start=config.warm_start,
        momentum=config.momentum,
        nesterovs_momentum=config.nesterovs_momentum,
        early_stopping=config.early_stopping,
        validation_fraction=config.validation_fraction,
        beta_1=config.beta_1,
        beta_2=config.beta_2,
        epsilon=config.epsilon,
        n_iter_no_change=config.n_iter_no_change,
        max_fun=config.max_fun,
    )

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
        explain_model(
            model,
            train_X,
            test_X,
            train_y,
            predictions,
            "kernel",
            shap_output_path,
        )


@dnn_app.command()
def regression(
    train_data_path: str = "example/data/regression_train.csv",
    test_data_path: str = "example/data/regression_test.csv",
    output_result_path: str = "example/data/regression_dnn_output.csv",
    shap_output_path: Optional[str] = typer.Option(
        None, help="Output folder for SHAP plot and values."
    ),
    preview_prediction_result: bool = False,
    label_name: str = "label",
    do_explain_model: bool = False,
    loss: Optional[Loss] = typer.Option(
        None,
        help="The loss function to use when training the weights. Default: squared_error",
        show_default=False,
    ),
    hidden_layer_sizes: Optional[str] = typer.Option(
        None,
        help="The ith element represents the number of neurons in the ith hidden layer. Default: (100,)",
        show_default=False,
    ),
    activation: Optional[Activation] = typer.Option(
        None,
        help="Activation function for the hidden layer. Default: relu",
        show_default=False,
    ),
    solver: Optional[Solver] = typer.Option(
        None,
        help="The solver for weight optimization. Default 'adam'",
        show_default=False,
    ),
    alpha: Optional[float] = typer.Option(
        None,
        help="Strength of the L2 regularization term. Default: 0.0001",
        show_default=False,
    ),
    batch_size: Optional[str] = typer.Option(
        None,
        help="Size of minibatches for stochastic optimizer. Default: 'auto'",
        show_default=False,
    ),
    learning_rate: Optional[LearningRate] = typer.Option(
        None,
        help="Learning rate schedule for weight updates Default: 'constant'",
        show_default=False,
    ),
    learning_rate_init: Optional[float] = typer.Option(
        None, help="The initial learning rate used. Default: 0.001", show_default=False
    ),
    power_t: Optional[float] = typer.Option(
        None,
        help="The exponent for inverse scaling learning rate. Default: 0.5",
        show_default=False,
    ),
    max_iter: Optional[int] = typer.Option(
        None, help="Maximum number of iterations. Default: 200", show_default=False
    ),
    shuffle: Optional[bool] = typer.Option(
        None,
        help="Whether to shuffle samples in each iteration. Default: True",
        show_default=True,
    ),
    random_state: Optional[int] = typer.Option(
        None,
        help="Determines random number generation. Default: None",
        show_default=False,
    ),
    tol: Optional[float] = typer.Option(
        None, help="Tolerance for the optimization. Default: 1e-4", show_default=False
    ),
    verbose: Optional[bool] = typer.Option(
        None,
        help="Whether to print progress messags to stdout. Default: False",
        show_default=False,
    ),
    warm_start: Optional[bool] = typer.Option(
        None,
        help="Whether reuse the solution of the previous call to fit as initialization. Default: False",
        show_default=False,
    ),
    momentum: Optional[float] = typer.Option(
        None,
        help="Momentum for gradient descent update. Default: 0.9",
        show_default=False,
    ),
    nesterovs_momentum: Optional[bool] = typer.Option(
        None,
        help="Whether to use Nesterov's Momentum. Default: True",
        show_default=False,
    ),
    early_stopping: Optional[bool] = typer.Option(
        None,
        help="Whether to use early stopping to terminate training when validation score is not improving. Default: False",
        show_default=False,
    ),
    validation_fraction: Optional[float] = typer.Option(
        None,
        help="The proportion of training data to set aside as validation set for early stopping. Default: 0.1",
        show_default=False,
    ),
    beta_1: Optional[float] = typer.Option(
        None,
        help="Exponential decay rate for estimates of first moment vector in adam. Default: 0.9",
        show_default=False,
    ),
    beta_2: Optional[float] = typer.Option(
        None,
        help="Exponential decay rate for estimates of second moment vector in adam. Default: 0.999",
        show_default=False,
    ),
    epsilon: Optional[float] = typer.Option(
        None,
        help="Value for numerical stability in adam. Default: 1e-8",
        show_default=False,
    ),
    n_iter_no_change: Optional[int] = typer.Option(
        None,
        help="Maximum number of epochs to not meet tol improvement. Default: 10",
        show_default=False,
    ),
    max_fun: Optional[int] = typer.Option(
        None,
        help="Maximum number of function calls. Only used when solver='lbfgs'. Default: 15000",
        show_default=False,
    ),
):

    hidden_layer_sizes = coerce_value(
        Config.__annotations__["hidden_layer_sizes"], hidden_layer_sizes
    )
    batch_size = coerce_value(Config.__annotations__["batch_size"], batch_size)

    config = build_config(
        loss=loss,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        power_t=power_t,
        max_iter=max_iter,
        shuffle=shuffle,
        random_state=random_state,
        tol=tol,
        verbose=verbose,
        warm_start=warm_start,
        momentum=momentum,
        nesterovs_momentum=nesterovs_momentum,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        n_iter_no_change=n_iter_no_change,
        max_fun=max_fun,
    )
    model = neural_network.MLPRegressor(
        loss=config.loss,
        hidden_layer_sizes=config.hidden_layer_sizes,
        activation=config.activation,
        solver=config.solver,
        alpha=config.alpha,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        learning_rate_init=config.learning_rate_init,
        power_t=config.power_t,
        max_iter=config.max_iter,
        shuffle=config.shuffle,
        random_state=config.random_state,
        tol=config.tol,
        verbose=config.verbose,
        warm_start=config.warm_start,
        momentum=config.momentum,
        nesterovs_momentum=config.nesterovs_momentum,
        early_stopping=config.early_stopping,
        validation_fraction=config.validation_fraction,
        beta_1=config.beta_1,
        beta_2=config.beta_2,
        epsilon=config.epsilon,
        n_iter_no_change=config.n_iter_no_change,
        max_fun=config.max_fun,
    )

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
        explain_model(
            model,
            train_X,
            test_X,
            train_y,
            predictions,
            "kernel",
            shap_output_path,
        )
