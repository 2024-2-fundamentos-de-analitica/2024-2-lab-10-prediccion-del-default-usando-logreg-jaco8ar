import pandas as pd
import numpy as np
import zipfile

import json
import os
import joblib
import gzip
from glob import glob

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, make_scorer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def read_zip_data(type_of_data):
    zip_path = f"files/input/{type_of_data}_data.csv.zip"
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        file_names = zip_file.namelist()
        with zip_file.open(file_names[0]) as file:
            file_df = pd.read_csv(file)
    return file_df

def clean_data(df):
    cleaned_df = df.copy()

    cleaned_df = cleaned_df.rename(columns = {"default payment next month": "default"})
    cleaned_df = cleaned_df.drop(columns = "ID")
    cleaned_df = cleaned_df.loc[cleaned_df["MARRIAGE"] != 0]
    cleaned_df = cleaned_df.loc[cleaned_df["EDUCATION"] != 0]
    cleaned_df["EDUCATION"] = cleaned_df["EDUCATION"].apply(lambda x: x if x < 4 else 4)
    
    return cleaned_df

    

def make_pipeline_logistic(categorical_features, numerical_features, k_best_features):
    # One-Hot Encoding for categorical variables
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # Min-Max Scaling for numerical variables
    numerical_transformer = MinMaxScaler()

    # Feature Preprocessing
    transformers = []
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))
    if numerical_features:
        transformers.append(("num", numerical_transformer, numerical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Ensure k_best_features does not exceed available features
    k_best_features = min(k_best_features, len(categorical_features) + len(numerical_features))

    # Create Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),  # Step 1: Preprocessing (One-Hot + Scaling)
            ("feature_selection", SelectKBest(score_func=f_classif, k=k_best_features)),  # Step 2: Feature Selection
            ("classifier", LogisticRegression(max_iter=500,random_state=42))  # Step 3: Logistic Regression Model
        ]
    )

    return pipeline


def optimize_pipeline(pipeline, X_train, y_train):

    param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],  
    "classifier__penalty": ["l1", "l2"], 
    "classifier__solver": ["liblinear", "saga"], 
    }


    scorer = make_scorer(balanced_accuracy_score)


    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        scoring=scorer, 
        cv=10, 
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    return  grid_search, grid_search.best_estimator_, grid_search.best_params_
    
import shutil
def create_output_directory(output_directory):
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

def save_model(path, model):
    create_output_directory("files/models/")

    with gzip.open(path, "wb") as f:
        joblib.dump(model, f)

    print(f"Model saved successfully at {path}")


def evaluate_model(model, X, y, dataset_name):

    y_pred = model.predict(X)

    metrics = {
        
        "type" : "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y, y_pred, average="weighted"),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "recall": recall_score(y, y_pred, average="weighted"),
        "f1_score": f1_score(y, y_pred, average="weighted"),
    }
    
    return metrics

def compute_confusion_matrix(model, X, y, dataset_name):
    """
    Computes the confusion matrix and returns it as a dictionary.
    """
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    cm_dict = {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {
            "predicted_0": int(cm[0, 0]), 
            "predicted_1": int(cm[0, 1])
        },
        "true_1": {
            "predicted_0": int(cm[1, 0]), 
            "predicted_1": int(cm[1, 1])
        },
    }

    return cm_dict
def run_job():

    train_data = read_zip_data("train")
    test_data = read_zip_data("test")
    train_data_clean = clean_data(train_data)
    test_data_clean = clean_data(test_data)


    X_train = train_data_clean.drop("default", axis = 1)
    X_test = test_data_clean.drop("default", axis = 1)

    y_train = train_data_clean["default"]
    y_test = test_data_clean["default"] 

    categorical_features = ["SEX","EDUCATION", "MARRIAGE"]
    numerical_features = [x for x in X_train.columns if x not in categorical_features]

    k_best = 10

    pipeline = make_pipeline_logistic(categorical_features, numerical_features, k_best)


    grid_search, best_model, best_params = optimize_pipeline(pipeline, X_train, y_train)

    save_model(
        os.path.join("files/models/", "model.pkl.gz"),
        grid_search
    )



   

    train_cm = compute_confusion_matrix(best_model, X_train, y_train, "train")
    test_cm = compute_confusion_matrix(best_model, X_test, y_test, "test")

    # Add "type" field to confusion matrices
    train_cm["type"] = "cm_matrix"
    test_cm["type"] = "cm_matrix"

    metrics = []

    train_metrics = evaluate_model(best_model, X_train, y_train, "train")
    test_metrics = evaluate_model(best_model, X_test, y_test, "test")
    
    metrics.append(train_metrics)
    metrics.append(test_metrics)
    # Append new confusion matrices
    metrics.append(train_cm)
    metrics.append(test_cm)

    os.makedirs("files/output/", exist_ok=True)
    with open("files/output/metrics.json", "w") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + "\n")

if __name__ == "__main__":
    run_job()
