import inspect
import uvicorn
from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from typing import Optional, Union, get_origin, get_args
from enum import Enum
from pathlib import Path

from src.svm import regression_logic as svm_regression, classification_logic as svm_classification, Loss, RegressionLoss, Kernel
from src.dnn import regression_logic as dnn_regression, classification_logic as dnn_classification

app = FastAPI()

# Tell the generator which params are file uploads
FILE_PARAMS = {"train_data_path", "test_data_path"}

def generate_form(func):
    sig = inspect.signature(func)
    form_html = "<form action='/run' method='post' enctype='multipart/form-data'>\n"
    
    for name, param in sig.parameters.items():
        ann = param.annotation
        default = param.default if param.default != inspect.Parameter.empty else ""
        help_text = getattr(default, "help", "") if hasattr(default, "help") else ""
        
        # Resolve actual default value for Typer.Option
        if hasattr(default, "default"):
            default_val = default.default
        else:
            default_val = default

        field_html = f"<label>{name}</label><br>"

        # File upload fields
        if name in FILE_PARAMS:
            field_html += f"<input type='file' name='{name}'><br><br>"
        
        # Bool checkbox
        elif ann == bool or isinstance(default_val, bool):
            checked = "checked" if default_val else ""
            field_html += f"<input type='checkbox' name='{name}' {checked}><br><br>"
        
        # Enum dropdown
        elif isinstance(ann, type) and issubclass(ann, Enum):
            field_html += f"<select name='{name}'>"
            for e in ann:
                selected = "selected" if e == default_val else ""
                field_html += f"<option value='{e.name}' {selected}>{e.name}</option>"
            field_html += "</select><br><br>"
        
        # Numeric
        elif ann in (int, float) or isinstance(default_val, (int, float)):
            step = "0.01" if ann == float or isinstance(default_val, float) else "1"
            val_str = f"value='{default_val}'" if default_val != inspect.Parameter.empty else ""
            field_html += f"<input type='number' name='{name}' step='{step}' {val_str}><br><br>"
        
        # String text
        else:
            val_str = f"value='{default_val}'" if default_val not in (inspect.Parameter.empty, None) else ""
            field_html += f"<input type='text' name='{name}' {val_str}><br><br>"
        
        form_html += field_html
    
    form_html += "<input type='submit' value='Run'>\n</form>"
    return form_html

@app.get("/", response_class=HTMLResponse)
def home():
    form_html = generate_form(regression_logic)
    return f"<h1>Run Regression</h1>{form_html}"

@app.post("/run")
async def run(request: Request,
              train_data_path: UploadFile = File(...),
              test_data_path: UploadFile = File(...),
              label_name: str = Form(...),
              svm_type: str = Form(...),
              C: Optional[float] = Form(None),
              kernel: str = Form("rbf"),
              preview_prediction_result: Optional[bool] = Form(False),
              do_explain_model: Optional[bool] = Form(False),
):
    # Save uploaded files
    train_path = Path(f"tmp_{train_data_path.filename}")
    with open(train_path, "wb") as f:
        f.write(await train_data_path.read())

    test_path = Path(f"tmp_{test_data_path.filename}")
    with open(test_path, "wb") as f:
        f.write(await test_data_path.read())

    output_path = Path("output.csv")

    regression_logic(
        train_data_path=str(train_path),
        test_data_path=str(test_path),
        output_result_path=str(output_path),
        shap_output_path=None,
        preview_prediction_result=preview_prediction_result,
        label_name=label_name,
        do_explain_model=do_explain_model,
        svm_type=SVMType[svm_type],
        C=C,
        nu=None,
        loss=RegressionLoss.epsilon_insensitive,
        dual=None,
        fit_intercept=None,
        intercept_scalling=None,
        kernel=Kernel[kernel],
        degree=None,
        gamma=None,
        coef0=None,
        shrinking=None,
        tol=None,
        cache_size=None,
        verbose=None,
        max_iter=None,
        epsilon=None,
    )

    return FileResponse(output_path, filename="result.csv")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

