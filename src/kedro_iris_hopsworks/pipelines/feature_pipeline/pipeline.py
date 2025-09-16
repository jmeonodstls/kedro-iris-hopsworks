"""
This is a boilerplate pipeline 'feature_pipeline'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_iris_data_node, insert_into_hopsworks_node

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=generate_iris_data_node,
            inputs={
                "BACKFILL": "params:BACKFILL",
                "iris_data": "iris_data"  # viene del catalog.yml
            },
            outputs="iris_df",
            name="generate_iris_data_node"
        ),
        node(
            func=insert_into_hopsworks_node,
            inputs={
                "df": "iris_df",
                "parameters": "params:iris_feature_group"
            },
            outputs=None,
            name="insert_into_hopsworks_node"
        )
    ])

