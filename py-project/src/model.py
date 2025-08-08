from sklearn import svm
import typer
from enum import Enum
from .utils import load_data

model_app = typer.Typer()

class SVMType(str, Enum):
    C = "C"
    Nu = "Nu"
    Linear = "Linear"

@model_app.command()
def classification(
    svm_type: SVMType = "C",
    train_data_path: str = "example/data/classification_train.csv",
    test_data_path: str = "example/data/classification_test.csv",
    label_name: str = "label"
):
    if svm_type == SVMType.C:
        model = svm.SVC()

    train_X, train_y = load_data(train_data_path, label_name)
    model.fit(train_X, train_y)

    test_X = load_data(test_data_path)
    test_pred = model.predict(test_X)
    print(test_pred)

