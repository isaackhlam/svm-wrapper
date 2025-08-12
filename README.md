# SVM-Wrapper

This program is a simple wrapper for SVM model. The implementation is adapted from `scikit-learn`.

## Usage

You are suggested to create a separate environment with conda first.

```sh
cd py-project
conda create -f environment.yml -n <env_name>
```

### Data Preparation

You need to prepare two csv files, `train.csv` and `test.csv`. Make sure the order of columns of two csv is same. And there is one more columns in the `train.csv`, namely `label`.  
There is example files in `example/data/*.csv`.

### Basic usage


Run the following command to perform SVM classification on example training data.  
That is `example/data/classification_train.csv` for training, and `example/data/classification_test.csv` for testing.  Prediction result with all feature will be stored in `example/data/classification_output.csv`

```sh
python main.py svm classification
```


The default SVM model is C-Support. You can specify C/Nu/Linear with `--svm-type`

```sh
python main.py svm classification --svm-type Nu 
```

You can also specify the input file and the label columns.

```sh
python main.py svm classification --train-data-path example/data/classification.csv
```

All supported parameters by the `svm` module from `scikit-learn` are supported here.  
You can run `python main.py svm classification --help` to see the full list.  
Extra not supported will be ignored

```sh
python main.py svm classification --svm-type Nu --nu 0.7 --c 1.2
```

## Development

To develop the program, run `make` under top directory (directory that contains this README). This will install all required development dependency and git hooks.
