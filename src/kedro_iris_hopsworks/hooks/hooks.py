# src/kedro_iris_hopsworks/hooks/hooks.py

from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog, MemoryDataset
from dotenv import load_dotenv
import os
import hopsworks

# Carga variables del .env
load_dotenv()

class HopsworksLoginHooks:
    @hook_impl
    def after_catalog_created(self, catalog: DataCatalog) -> None:
        """Login en Hopsworks antes de ejecutar cualquier pipeline."""
        project = hopsworks.login(
            project="first_ml_system",
            api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        )
        # Inyectar el proyecto autenticado como un dataset en memoria
        catalog.add("project_hopsworks", MemoryDataset(data=project))