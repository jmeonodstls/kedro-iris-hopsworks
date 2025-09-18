"""
This is a boilerplate pipeline 'inference_pipeline'
generated using Kedro 1.0.0
"""

import os
import joblib
from PIL import Image
from datetime import datetime
import shutil
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from dotenv import load_dotenv

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

from hsml.schema import Schema
from hsml.model_schema import ModelSchema

import hopsworks

import time

def download_model(project, params):
    mr = project.get_model_registry()
    model = mr.get_model(params["model_name"], version=params["model_version"])
    model_dir = model.download()
    model = joblib.load(f"{model_dir}/iris_model.pkl")
    return model

def load_batch_data(project, params):
    fs = project.get_feature_store()
    fv = fs.get_feature_view(
        name=params["feature_view_name"],
        version=params["feature_view_version"]
    )
    return fv.get_batch_data()

def run_inference(trained_model, batch_data):
    y_pred = trained_model.predict(batch_data)
    return y_pred

def save_predicted_image(y_pred):
    flower = y_pred[-1]
    flower_img = f"data/08_reporting/{flower}.png"
    #original_path = f"data/08_reporting/{flower}.png"

    img = Image.open(flower_img)
    img.save("data/08_reporting/latest_iris.png")

    return flower

def save_actual_image(project, params):
    """
    Carga el último label real desde el Feature Group y guarda su imagen correspondiente.
    Retorna el label real para ser usado en otros nodos.
    """
    fs = project.get_feature_store()

    iris_fg = fs.get_feature_group(
        name=params["name"],
        version=params["version"]
    )
    df = iris_fg.read()

    label = df.iloc[-1]["variety"]

    # Ruta a la imagen correspondiente al label real
    label_flower = f"data/08_reporting/{label}.png"

    # Carga y guarda la imagen con nuevo nombre
    img = Image.open(label_flower)
    img.save("data/08_reporting/actual_iris.png")

    return label


def save_predictions(project, flower, label, params):
    # 1. Obtener el Feature Group
    fs = project.get_feature_store()
    monitor_fg = fs.get_or_create_feature_group(
        name=params["prediction_fg_name"],
        version=params["prediction_fg_version"],
        primary_key=[params["prediction_fg_primary_key"]],
        description="Iris prediction monitoring by Kedro"
    )

    # 2. Insertar nueva fila de predicción
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    monitor_df = pd.DataFrame({
        "prediction": [flower],
        "label": [label],
        "datetime": [now]
    })

    monitor_fg.insert(monitor_df)
    
    # Esto lo puedo separar y convertirlo en otro nodo hasta esperar que los datos esten escritos en hopsworks
    # 3. Leer todo el histórico
    time.sleep(120)
    history_df = monitor_fg.read()

    # 4. Guardar como Excel en data/08_reporting
    os.makedirs("data/08_reporting", exist_ok=True)
    history_df.to_excel("data/08_reporting/history_predictions.xlsx", index=False)

    return history_df
