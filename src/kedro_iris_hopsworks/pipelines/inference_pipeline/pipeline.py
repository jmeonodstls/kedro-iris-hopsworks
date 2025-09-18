"""
This is a boilerplate pipeline 'inference_pipeline'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Pipeline, node, pipeline
from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=nodes.download_model,
            inputs=["project_hopsworks", "params:inference"],
            outputs="inference_model",
            name="download_model_node"
        ),
        node(
            func=nodes.load_batch_data,
            inputs=["project_hopsworks", "params:inference"],
            outputs="inference_batch_data",
            name="load_batch_data_node"
        ),
        node(
            func=nodes.run_inference,
            inputs=["inference_model", "inference_batch_data"],
            outputs="y_pred",
            name="run_inference_node"
        ),
        node(
            func=nodes.save_predicted_image,
            inputs="y_pred",
            outputs="inference_flower",
            name="save_predicted_image_node"
        ),
        node(
            func=nodes.save_actual_image,
            inputs=["project_hopsworks", "params:iris_feature_group"],
            outputs="inference_label",
            name="save_actual_image_node",
        ),
        node(
            func=nodes.save_predictions,
            inputs=["project_hopsworks", "inference_flower", "inference_label", "params:inference"],
            outputs="history_predictions",
            name="save_predictions_node"
        ),
    ])

