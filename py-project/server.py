import inspect
import uvicorn
from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from typing import Optional, Union, get_origin, get_args
from enum import Enum
from pathlib import Path

from src.svm import (
    regression_logic as svm_regression,
    classification_logic as svm_classification,
    Loss,
    RegressionLoss,
    Kernel,
    SVMType,
    DecisionFunctionShape,
    Penalty,
    MultiClass,
    coerce_value,
)
from src.dnn import (
    regression_logic as dnn_regression,
    classification_logic as dnn_classification,
    Activation,
    Solver,
    LearningRate,
    Config,
    Loss as DNNLoss
)

app = FastAPI()

# Tell the generator which params are file uploads
FILE_PARAMS = {"train_data_path", "test_data_path"}


def unwrap_optional_enum(ann):
    origin = get_origin(ann)
    if origin is Union:  # Optional[T] is Union[T, NoneType]
        args = [a for a in get_args(ann) if a is not type(None)]
        if len(args) == 1 and isinstance(args[0], type) and issubclass(args[0], Enum):
            return args[0]
    return ann


def generate_form(func, action_url: str, title: str):
    sig = inspect.signature(func)
    form_html = f"<h1>{title}</h1>\n"
    form_html += (
        f"<form action='{action_url}' method='post' enctype='multipart/form-data'>\n"
    )

    for name, param in sig.parameters.items():
        ann = unwrap_optional_enum(param.annotation)
        default = param.default if param.default != inspect.Parameter.empty else ""

        # Resolve Typer.Option default value if present
        if hasattr(default, "default"):
            default_val = default.default
        else:
            default_val = default

        field_html = f"<label>{name}</label><br>"

        if name in FILE_PARAMS:
            field_html += f"<input type='file' name='{name}'><br><br>"
        elif isinstance(ann, type) and issubclass(ann, Enum):
            field_html += f"<select name='{name}'>"
            for e in ann:
                selected = "selected" if e == default_val else ""
                field_html += f"<option value='{e.name}' {selected}>{e.name}</option>"
            field_html += "</select><br><br>"
        elif ann == bool or isinstance(default_val, bool):
            checked = "checked" if default_val else ""
            field_html += f"<input type='checkbox' name='{name}' {checked}><br><br>"
        elif ann in (int, float) or isinstance(default_val, (int, float)):
            step = "0.01" if ann == float or isinstance(default_val, float) else "1"
            val_str = (
                f"value='{default_val}'"
                if default_val != inspect.Parameter.empty
                else ""
            )
            field_html += (
                f"<input type='number' step='{step}' name='{name}' {val_str}><br><br>"
            )
        else:
            val_str = (
                f"value='{default_val}'"
                if default_val not in (inspect.Parameter.empty, None)
                else ""
            )
            field_html += f"<input type='text' name='{name}' {val_str}><br><br>"

        form_html += field_html

    form_html += "<input type='submit' value='Run'>\n</form>"
    return form_html


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>Model Runner</h1>
    <a href='/svm/reg/'>SVM Regression</a><br>
    <a href='/svm/cls/'>SVM Classification</a><br>
    <a href='/dnn/reg/'>DNN Regression</a><br>
    <a href='/dnn/cls/'>DNN Classification</a><br>
    """


@app.get("/svm/reg/", response_class=HTMLResponse)
def svm_reg_form():
    return generate_form(svm_regression, "/svm/reg/run", "SVM Regression")


@app.post("/svm/reg/run")
async def svm_reg_run(
    request: Request,
    train_data_path: UploadFile = File(...),
    test_data_path: UploadFile = File(...),
    label_name: str = Form("label"),
    svm_type: str = Form(...),
    C: Optional[float] = Form(1.0),
    kernel: str = Form("rbf"),
    preview_prediction_result: Optional[bool] = Form(False),
    do_explain_model: Optional[bool] = Form(False),
    nu: Optional[float] = Form(0.5),
    loss: Optional[RegressionLoss] = Form("epsilon_insensitive"),
    dual: Optional[Union[str, bool]] = Form("auto"),
    fit_intercept: Optional[bool] = Form(True),
    intercept_scalling: Optional[float] = Form(1.0),
    degree: Optional[int] = Form(3),
    gamma: Optional[Union[float, str]] = Form("scale"),
    coef0: Optional[float] = Form(0.0),
    shrinking: Optional[bool] = Form(True),
    tol: Optional[float] = Form(1e-3),
    cache_size: Optional[float] = Form(200),
    verbose: Optional[Union[bool, int]] = Form(False),
    max_iter: Optional[int] = Form(-1),
    epsilon: float = Form(0.0),
):

    # Save uploaded files
    train_path = Path(f"tmp_{train_data_path.filename}")
    with open(train_path, "wb") as f:
        f.write(await train_data_path.read())

    test_path = Path(f"tmp_{test_data_path.filename}")
    with open(test_path, "wb") as f:
        f.write(await test_data_path.read())

    output_path = Path("output.csv")

    svm_regression(
        train_data_path=str(train_path),
        test_data_path=str(test_path),
        output_result_path=str(output_path),
        shap_output_path=None,
        preview_prediction_result=preview_prediction_result,
        label_name=label_name,
        do_explain_model=do_explain_model,
        svm_type=SVMType[svm_type],
        C=C,
        nu=nu,
        loss=RegressionLoss[loss],
        dual=dual,
        fit_intercept=fit_intercept,
        intercept_scalling=intercept_scalling,
        kernel=Kernel[kernel],
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

    return FileResponse(output_path, filename="result.csv")


# Repeat the same pattern for svm_cls_logic, dnn
@app.get("/svm/cls/", response_class=HTMLResponse)
def svm_cls_form():
    return generate_form(svm_classification, "/svm/cls/run", "SVM Classification")


@app.post("/svm/cls/run")
async def svm_cls_run(
    request: Request,
    train_data_path: UploadFile = File(...),
    test_data_path: UploadFile = File(...),
    label_name: str = Form("label"),
    svm_type: str = Form(...),
    C: Optional[float] = Form(1.0),
    kernel: str = Form("rbf"),
    preview_prediction_result: Optional[bool] = Form(False),
    do_explain_model: Optional[bool] = Form(False),
    nu: Optional[float] = Form(0.5),
    penalty: Optional[Penalty] = Form(None),
    loss: Optional[Loss] = Form("squared_hinge"),
    dual: Optional[Union[str, bool]] = Form("auto"),
    multi_class: Optional[MultiClass] = Form("ovr"),
    fit_intercept: Optional[bool] = Form(True),
    intercept_scalling: Optional[float] = Form(1.0),
    degree: Optional[int] = Form(3),
    gamma: Optional[Union[float, str]] = Form("scale"),
    coef0: Optional[float] = Form(0.0),
    shrinking: Optional[bool] = Form(True),
    probability: Optional[bool] = Form(False),
    tol: Optional[float] = Form(1e-3),
    cache_size: Optional[float] = Form(200),
    class_weight: Optional[Union[dict, str]] = Form(None),
    verbose: Optional[Union[bool, int]] = Form(False),
    max_iter: Optional[int] = Form(-1),
    decision_function_shape: Optional[DecisionFunctionShape] = Form("ovr"),
    break_ties: Optional[bool] = Form(False),
    random_state: Optional[int] = Form(42),
):

    # Save uploaded files
    train_path = Path(f"tmp_{train_data_path.filename}")
    with open(train_path, "wb") as f:
        f.write(await train_data_path.read())

    test_path = Path(f"tmp_{test_data_path.filename}")
    with open(test_path, "wb") as f:
        f.write(await test_data_path.read())

    output_path = Path("output.csv")

    svm_classification(
        train_data_path=str(train_path),
        test_data_path=str(test_path),
        output_result_path=str(output_path),
        shap_output_path=None,
        preview_prediction_result=preview_prediction_result,
        label_name=label_name,
        do_explain_model=do_explain_model,
        svm_type=SVMType[svm_type],
        C=C,
        nu=nu,
        penalty=penalty,
        loss=Loss[loss],
        dual=dual,
        multi_class=multi_class,
        fit_intercept=fit_intercept,
        intercept_scalling=intercept_scalling,
        kernel=Kernel[kernel],
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

    return FileResponse(output_path, filename="result.csv")


# Repeat the same pattern for svm_cls_logic, dnn
@app.get("/dnn/cls/", response_class=HTMLResponse)
def dnn_cls_form():
    return generate_form(dnn_classification, "/dnn/cls/run", "DNN Classification")


@app.post("/dnn/cls/run")
async def dnn_cls_run(
    request: Request,
    train_data_path: UploadFile = File(...),
    test_data_path: UploadFile = File(...),
    label_name: str = Form("label"),
    preview_prediction_result: bool = Form(False),
    do_explain_model: bool = Form(False),
    hidden_layer_sizes: Optional[str] = Form("(100,)"),
    activation: Optional[Activation] = Form("relu"),
    solver: Optional[Solver] = Form("adam"),
    alpha: Optional[float] = Form(0.0001),
    batch_size: Optional[Union[str, float]] = Form("auto"),
    learning_rate: Optional[LearningRate] = Form("constant"),
    learning_rate_init: Optional[float] = Form(0.001),
    power_t: Optional[float] = Form(0.5),
    max_iter: Optional[int] = Form(200),
    shuffle: Optional[bool] = Form(True),
    random_state: Optional[int] = Form(42),
    tol: Optional[float] = Form(1e-4),
    verbose: Optional[bool] = Form(False),
    warm_start: Optional[bool] = Form(False),
    momentum: Optional[float] = Form(0.9),
    nesterovs_momentum: Optional[bool] = Form(True),
    early_stopping: Optional[bool] = Form(False),
    validation_fraction: Optional[float] = Form(0.1),
    beta_1: Optional[float] = Form(0.9),
    beta_2: Optional[float] = Form(0.999),
    epsilon: Optional[float] = Form(1e-8),
    n_iter_no_change: Optional[int] = Form(10),
    max_fun: Optional[int] = Form(15000),
):

    # Save uploaded files
    hidden_layer_sizes = coerce_value(
        Config.__annotations__["hidden_layer_sizes"], hidden_layer_sizes
    )
    train_path = Path(f"tmp_{train_data_path.filename}")
    with open(train_path, "wb") as f:
        f.write(await train_data_path.read())

    test_path = Path(f"tmp_{test_data_path.filename}")
    with open(test_path, "wb") as f:
        f.write(await test_data_path.read())

    output_path = Path("output.csv")

    dnn_classification(
        train_data_path=str(train_path),
        test_data_path=str(test_path),
        output_result_path=str(output_path),
        shap_output_path=None,
        preview_prediction_result=preview_prediction_result,
        label_name=label_name,
        do_explain_model=do_explain_model,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=Activation[activation],
        solver=Solver[solver],
        alpha=alpha,
        batch_size=batch_size,
        learning_rate=LearningRate[learning_rate],
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

    return FileResponse(output_path, filename="result.csv")

# Repeat the same pattern for svm_cls_logic, dnn
@app.get("/dnn/reg/", response_class=HTMLResponse)
def dnn_reg_form():
    return generate_form(dnn_classification, "/dnn/reg/run", "DNN Regression")


@app.post("/dnn/reg/run")
async def dnn_reg_run(
    request: Request,
    train_data_path: UploadFile = File(...),
    test_data_path: UploadFile = File(...),
    label_name: str = Form("label"),
    preview_prediction_result: bool = Form(False),
    do_explain_model: bool = Form(False),
    hidden_layer_sizes: Optional[str] = Form("(100,)"),
    activation: Optional[Activation] = Form("relu"),
    solver: Optional[Solver] = Form("adam"),
    alpha: Optional[float] = Form(0.0001),
    batch_size: Optional[Union[str, float]] = Form("auto"),
    learning_rate: Optional[LearningRate] = Form("constant"),
    learning_rate_init: Optional[float] = Form(0.001),
    power_t: Optional[float] = Form(0.5),
    loss: Optional[DNNLoss] = Form("squared_error"),
    max_iter: Optional[int] = Form(200),
    shuffle: Optional[bool] = Form(True),
    random_state: Optional[int] = Form(42),
    tol: Optional[float] = Form(1e-4),
    verbose: Optional[bool] = Form(False),
    warm_start: Optional[bool] = Form(False),
    momentum: Optional[float] = Form(0.9),
    nesterovs_momentum: Optional[bool] = Form(True),
    early_stopping: Optional[bool] = Form(False),
    validation_fraction: Optional[float] = Form(0.1),
    beta_1: Optional[float] = Form(0.9),
    beta_2: Optional[float] = Form(0.999),
    epsilon: Optional[float] = Form(1e-8),
    n_iter_no_change: Optional[int] = Form(10),
    max_fun: Optional[int] = Form(15000),
):

    # Save uploaded files
    hidden_layer_sizes = coerce_value(
        Config.__annotations__["hidden_layer_sizes"], hidden_layer_sizes
    )
    train_path = Path(f"tmp_{train_data_path.filename}")
    with open(train_path, "wb") as f:
        f.write(await train_data_path.read())

    test_path = Path(f"tmp_{test_data_path.filename}")
    with open(test_path, "wb") as f:
        f.write(await test_data_path.read())

    output_path = Path("output.csv")

    dnn_regression(
        train_data_path=str(train_path),
        test_data_path=str(test_path),
        output_result_path=str(output_path),
        shap_output_path=None,
        preview_prediction_result=preview_prediction_result,
        label_name=label_name,
        do_explain_model=do_explain_model,
        loss=DNNLoss[loss],
        hidden_layer_sizes=hidden_layer_sizes,
        activation=Activation[activation],
        solver=Solver[solver],
        alpha=alpha,
        batch_size=batch_size,
        learning_rate=LearningRate[learning_rate],
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

    return FileResponse(output_path, filename="result.csv")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
