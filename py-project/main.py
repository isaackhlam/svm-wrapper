import sys

import typer
from src.svm import svm_app
from src.dnn import dnn_app
from src.utils import setup_logger

app = typer.Typer()
app.add_typer(svm_app, name="svm")
app.add_typer(dnn_app, name="dnn")


@app.command()
def hello(name: str):
    print(f"Hello {name}")


@app.command()
def goodbye(name: str, formal: bool = False):
    if formal:
        print(f"Goodbye Ms. {name}. Have a good day.")
    else:
        print(f"Bye {name}!")


if __name__ == "__main__":
    logger = setup_logger("app.log")
    app()
