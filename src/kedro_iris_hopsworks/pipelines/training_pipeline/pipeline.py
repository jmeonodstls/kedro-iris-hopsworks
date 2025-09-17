"""
This is a boilerplate pipeline 'training_pipeline'
generated using Kedro 1.0.0
"""

# src/kedro_iris_hopsworks/pipelines/training_pipeline/pipeline.py

from kedro.pipeline import Pipeline, node, pipeline
from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.load_feature_view,
                inputs=["project_hopsworks", "params:feature_view"],  # ✅ se agrega project
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="load_feature_view_node",
            ),
            node(
                func=nodes.train_model,
                inputs=["X_train", "y_train", "params:n_neighbors"],
                outputs="trained_model",
                name="train_model_node",
            ),
            node(
                func=nodes.evaluate_model,
                inputs=["trained_model", "X_test", "y_test"],
                outputs="metrics",
                name="evaluate_model_node",
            ),
            node(
                func=nodes.save_confusion_matrix,
                inputs=["trained_model", "X_test", "y_test"],
                outputs=None,
                name="save_confusion_matrix_node",
            ),
            node(
                func=nodes.register_model,
                inputs=[
                    "project_hopsworks",        # ✅ se agrega project aquí también
                    "trained_model",
                    "X_train",
                    "y_train",
                    "metrics",
                    "params:model",
                ],
                outputs=None,
                name="register_model_node",
            ),
        ]
    )
