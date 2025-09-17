"""
This is a boilerplate pipeline 'training_pipeline'
generated using Kedro 1.0.0
"""
# src/kedro_iris_hopsworks/pipelines/training_pipeline/nodes.py

import os
import joblib
import shutil
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from dotenv import load_dotenv

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

#from hsfs.client.exceptions import RestAPIError
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

import hopsworks

def load_feature_view(project, params):
    """Carga un Feature View si existe, o lo crea si no existe."""
    fs = project.get_feature_store()

    feature_view = None

    # Intentar obtener el feature view (puede lanzar excepción si no existe)
    try:
        feature_view = fs.get_feature_view(
            name=params["name"],
            version=params["version"]
        )
    except Exception:
        pass  # No existe o falló la consulta, seguimos para crearlo

    # Si no se pudo obtener, lo creamos desde el Feature Group
    if feature_view is None:
        fg = fs.get_feature_group(name=params["name"], version=params["version"])
        query = fg.select_all()

        feature_view = fs.create_feature_view(
            name=params["name"],
            version=params["version"],
            description="Created by Kedro pipeline",
            labels=[params["label"]],
            query=query,
        )

    # Dividir el feature view
    X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, n_neighbors):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train.values.ravel())
    return model


def evaluate_model(trained_model, X_test, y_test):
    y_pred = trained_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report


def save_confusion_matrix(trained_model, X_test, y_test):
    y_pred = trained_model.predict(X_test)
    results = confusion_matrix(y_test, y_pred)

    df_cm = pd.DataFrame(
        results,
        index=["True Setosa", "True Versicolor", "True Virginica"],
        columns=["Pred Setosa", "Pred Versicolor", "Pred Virginica"]
    )

    cm = sns.heatmap(df_cm, annot=True)
    fig = cm.get_figure()

    os.makedirs("data/08_reporting", exist_ok=True)
    fig.savefig("data/08_reporting/confusion_matrix.png")
    plt.close(fig)  # Evita mostrar en ejecución batch


def register_model(project, trained_model, X_train, y_train, metrics, model_params):
    mr = project.get_model_registry()

    model_dir = "iris_model"
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(trained_model, f"{model_dir}/iris_model.pkl")
    shutil.copyfile("data/08_reporting/confusion_matrix.png", f"{model_dir}/confusion_matrix.png")

    input_example = X_train.sample()
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema, output_schema)

    model = mr.python.create_model(
        version=model_params["version"],
        name=model_params["name"],
        description=model_params["description"],
        metrics={"accuracy": metrics["accuracy"]},
        input_example=input_example,
        model_schema=model_schema,
    )

    model.save(model_dir)
