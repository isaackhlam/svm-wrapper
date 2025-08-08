import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# Settings
n_samples = 100
n_features = 20
random_state = 42

# --- Classification Data ---
X_cls, y_cls = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=random_state,
)

# Split
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=random_state
)

# Create DataFrames
feature_names = [f"feature_{i+1}" for i in range(n_features)]

df_cls_train = pd.DataFrame(X_cls_train, columns=feature_names)
df_cls_train["label"] = y_cls_train

df_cls_test = pd.DataFrame(X_cls_test, columns=feature_names)
df_cls_test["label"] = y_cls_test

# --- Regression Data ---
X_reg, y_reg = make_regression(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=15,
    noise=0.1,
    random_state=random_state,
)

# Split
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=random_state
)

# Create DataFrames
df_reg_train = pd.DataFrame(X_reg_train, columns=feature_names)
df_reg_train["label"] = y_reg_train

df_reg_test = pd.DataFrame(X_reg_test, columns=feature_names)
df_reg_test["label"] = y_reg_test

# --- Save CSV files ---
df_cls_train.to_csv("classification_train.csv", index=False)
df_cls_test.to_csv("classification_test.csv", index=False)
df_reg_train.to_csv("regression_train.csv", index=False)
df_reg_test.to_csv("regression_test.csv", index=False)

print("✅ All 4 datasets generated and saved successfully.")
