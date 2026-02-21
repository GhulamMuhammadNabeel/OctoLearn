"""
Model Registry Module

Manages model versioning, metadata storage, and artifact retrieval.
Uses JSON as a backend and handles NumPy/Pandas serialization safely.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import copy
from datetime import datetime
from typing import Dict, Any, Optional

from ..config import MODEL_REGISTRY_CONFIG
from ..utils.helpers import setup_logger

logger = setup_logger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy and Pandas objects.

    This encoder converts NumPy numeric types, arrays, and Pandas timestamps
    into native Python types so they can be serialized into JSON.
    """

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        return super().default(obj)


class ModelRegistry:
    """
    Persistence layer for model artifacts and performance metadata.

    The registry manages versioned storage of fitted models (as joblib files)
    and their associated metadata (as a JSON database). It provides a centralized
    point for model tracking and lifecycle management.

    Attributes
    ----------
    db_path : str
        The absolute path to the JSON registry database.
    storage_dir : str
        The base directory where all artifacts and the database are stored.
    models_dir : str
        The subdirectory dedicated specifically to model pkl files.

    Notes
    -----
    Uses a standard directory structure:
    `artifact_dir/`
    ├── `model_registry.json`
    └── `trained_models/`
        └── `model_name_v1.pkl`
    """

    def __init__(self):
        """
        Initialize ModelRegistry and ensure required directories and files exist.
        """
        self.config = MODEL_REGISTRY_CONFIG
        self.db_path = self.config["db_path"]

        self.storage_dir = os.path.dirname(self.db_path)
        self.models_dir = os.path.join(self.storage_dir, "trained_models")

        self._initialize_registry()

    def _initialize_registry(self) -> None:
        """
        Ensure registry directories and JSON database file exist.

        Creates:
        - Storage directory
        - trained_models directory
        - JSON registry file (if missing)
        """
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        if not os.path.exists(self.db_path):
            with open(self.db_path, "w") as f:
                json.dump({"models": {}, "experiments": []}, f, indent=4)

    def register_model(
        self,
        name: str,
        model: Any,
        task_type: str,
        metrics: Dict[str, float],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        """
        Save a trained model and its performance metadata to the registry.

        This method assigns a version number, persists the model object to disk,
        and updates the JSON database with the provided metrics and parameters.

        Parameters
        ----------
        name : str
            The identifier for the model algorithm (e.g., 'xgboost').
        model : Any
            The fitted, scikit-learn compatible model object.
        task_type : str
            The nature of the problem ('classification' or 'regression').
        metrics : dict of str: float
            The performance scores for this specific model instance.
        parameters : dict, optional
            The hyperparameters used for training.

        Returns
        -------
        version : int, optional
            The assigned version number if successful, or None on failure.
        """
        try:
            registry_data = self._load_registry()

            timestamp = datetime.now().isoformat()
            version = len(registry_data["models"].get(name, [])) + 1

            model_filename = f"{name}_v{version}.pkl"
            model_path = os.path.join(self.models_dir, model_filename)

            # Save model artifact
            joblib.dump(model, model_path)

            entry = {
                "version": version,
                "timestamp": timestamp,
                "task_type": task_type,
                "path": model_path,
                "metrics": metrics,
                "parameters": parameters or {}
            }

            registry_data["models"].setdefault(name, []).append(entry)

            self._save_registry(registry_data)

            logger.info(f"Registered model: {name} (v{version})")
            return version

        except Exception as e:
            logger.error(f"Failed to register model {name}: {str(e)}")
            return None

    def get_best_model(
        self,
        metric: str = "score",
        mode: str = "max"
    ) -> Optional[Any]:
        """
        Retrieve the best performing model across all registered models.

        Parameters
        ----------
        metric : str
            Metric name to compare (default: 'score').
        mode : str
            'max' to maximize metric, 'min' to minimize metric.

        Returns
        -------
        Any or None
            Loaded model object if found, else None.
        """
        registry_data = self._load_registry()

        best_model_path = None
        best_value = -float("inf") if mode == "max" else float("inf")

        for name, versions in registry_data.get("models", {}).items():
            for entry in versions:
                val = entry.get("metrics", {}).get(metric)

                if val is None:
                    continue

                if (mode == "max" and val > best_value) or (mode == "min" and val < best_value):
                    best_value = val
                    best_model_path = entry["path"]

        if best_model_path and os.path.exists(best_model_path):
            logger.info(f"Loaded best model from {best_model_path}")
            return joblib.load(best_model_path)

        logger.warning("No suitable best model found.")
        return None

    def list_models(self) -> Dict[str, Any]:
        """
        List all registered models and their metadata.

        Returns
        -------
        Dict[str, Any]
            Dictionary of model names and their version metadata.
        """
        registry_data = self._load_registry()
        # FIX: return deep copy so callers cannot mutate internal registry state by accident
        return copy.deepcopy(registry_data.get("models", {}))

    def load_model(self, name: str, version: Optional[int] = None) -> Optional[Any]:
        """
        Load a specific model by name and version.

        Parameters
        ----------
        name : str
            Model name.
        version : int, optional
            Model version. If None, loads the latest version.

        Returns
        -------
        Any or None
            Loaded model object if found, else None.
        """
        registry_data = self._load_registry()

        if name not in registry_data.get("models", {}):
            logger.warning(f"Model {name} not found in registry.")
            return None

        versions = registry_data["models"][name]

        if version is None:
            entry = versions[-1]
        else:
            entry = next((v for v in versions if v["version"] == version), None)

        if entry and os.path.exists(entry["path"]):
            logger.info(f"Loaded model {name} v{entry['version']}")
            return joblib.load(entry["path"])

        logger.error(f"Model file not found for {name} v{version}")
        return None

    def _load_registry(self) -> Dict[str, Any]:
        """
        Load registry JSON file safely.

        Returns
        -------
        Dict[str, Any]
            Registry dictionary.
        """
        try:
            with open(self.db_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("Registry file corrupted or missing. Reinitializing.")
            return {"models": {}, "experiments": []}

    def _save_registry(self, data: Dict[str, Any]) -> None:
        """
        Save registry data to JSON file using NumpyEncoder.

        Parameters
        ----------
        data : Dict[str, Any]
            Registry data to save.
        """
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder)
