"""
This is a boilerplate pipeline 'feature_pipeline'
generated using Kedro 1.0.0
"""
### ✅ Nodo `generate_iris_data_node`

import random
import pandas as pd
import os
import hopsworks
from dotenv import load_dotenv

def generate_flower(name, sepal_len_min, sepal_len_max, sepal_width_min, sepal_width_max, 
                    petal_len_min, petal_len_max, petal_width_min, petal_width_max):
    df = pd.DataFrame({
        "sepal_length": [random.uniform(sepal_len_min, sepal_len_max)],
        "sepal_width": [random.uniform(sepal_width_min, sepal_width_max)],
        "petal_length": [random.uniform(petal_len_min, petal_len_max)],
        "petal_width": [random.uniform(petal_width_min, petal_width_max)]
    })
    df['variety'] = name
    return df


def get_random_iris_flower():
    virginica_df = generate_flower("Virginica", 5.5, 8, 2.2, 3.8, 4.5, 7, 1.4, 2.5)
    versicolor_df = generate_flower("Versicolor", 4.5, 7.5, 2.1, 3.5, 3.1, 5.5, 1.0, 1.8)
    setosa_df = generate_flower("Setosa", 4.5, 6, 2.3, 4.5, 1.2, 2, 0.3, 0.7)

    pick_random = random.uniform(0, 3)
    if pick_random >= 2:
        return virginica_df
    elif pick_random >= 1:
        return versicolor_df
    else:
        return setosa_df


def generate_iris_data_node(BACKFILL: bool, iris_data: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un DataFrame de iris aleatorio o lo toma del CSV, dependiendo del flag BACKFILL.
    """
    random.seed()
    if BACKFILL:
        return iris_data
    else:
        return get_random_iris_flower()

### ✅ Nodo `insert_into_hopsworks_node`
def insert_into_hopsworks_node(df, parameters):
    """
    Inserta los datos en el Feature Group de Hopsworks usando los parámetros especificados.
    """
    # Cargar variables de entorno
    load_dotenv()

    # Autenticación con API Key desde el archivo .env
    project = hopsworks.login(
        project="first_ml_system",  # nombre de tu proyecto en Hopsworks
        api_key_value=os.getenv("HOPSWORKS_API_KEY")
    )
    
    fs = project.get_feature_store()

    iris_fg = fs.get_or_create_feature_group(
        name=parameters["name"],
        version=parameters["version"],
        primary_key=parameters["primary_key"],
        description=parameters["description"]
    )

    iris_fg.insert(df, write_options={"ignore_duplicate_keys": "true", "wait_for_job": "true"})
    
    return df
