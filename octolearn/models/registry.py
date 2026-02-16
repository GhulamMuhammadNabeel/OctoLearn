"""
Model Registry Module

Manages model storage, versioning, and retrieval
"""

import json
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import warnings
import csv

warnings.filterwarnings('ignore')

# Try to import sqlite3, but don't fail if unavailable
try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

from ..config import MODEL_REGISTRY_CONFIG
from ..utils.helpers import setup_logger

logger = setup_logger(__name__)


class ModelRegistry:
    """
    Registry for storing and managing trained models, their metadata, and versioning.

    Supports JSON, SQLite, and CSV backends for model persistence and retrieval.

    Attributes:
        storage_type (str): Backend type ('json', 'sqlite', 'csv').
        db_path (str): Path to storage file or database.
    """
    
    def __init__(self, storage_type: str = None, db_path: str = None):
        """
        Initialize ModelRegistry with backend and storage path.

        Args:
            storage_type (str, optional): Storage backend: 'json' (default), 'sqlite', or 'csv'.
            db_path (str, optional): Path to storage database/file.
        """
        requested_storage = storage_type or MODEL_REGISTRY_CONFIG['storage']
        self.db_path = db_path or MODEL_REGISTRY_CONFIG['db_path']
        
        # Validate and set storage type with fallback
        self.storage_type = self._validate_storage_type(requested_storage)
        
        # Create storage directory if needed
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        if self.storage_type == 'sqlite':
            self._init_sqlite()
        elif self.storage_type == 'json':
            self._init_json()
        elif self.storage_type == 'csv':
            self._init_csv()
        
        logger.info(f"ModelRegistry initialized with {self.storage_type} storage at {self.db_path}")
    
    def _validate_storage_type(self, requested_type: str) -> str:
        """
        Validate storage type and fallback to JSON if needed.

        Args:
            requested_type (str): Requested storage type.

        Returns:
            str: Valid storage type to use.
        """
        if requested_type == 'sqlite':
            if not SQLITE_AVAILABLE:
                logger.warning(
                    "sqlite3 not available. Falling back to JSON storage. "
                    "Install python-dev for sqlite3 support if needed."
                )
                return 'json'
            return 'sqlite'
        
        elif requested_type == 'csv':
            return 'csv'
        
        elif requested_type == 'json':
            return 'json'
        
        else:
            logger.warning(f"Unknown storage type '{requested_type}'. Using JSON.")
            return 'json'
    
    def _init_sqlite(self):
        """Initialize SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    model_path TEXT NOT NULL,
                    task_type TEXT,
                    metrics TEXT,
                    parameters TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(name, version)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("SQLite database initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize SQLite: {str(e)}")
    
    def _init_json(self):
        """Initialize JSON storage."""
        try:
            if not Path(self.db_path).exists():
                with open(self.db_path, 'w') as f:
                    json.dump({}, f, indent=2)
            
            logger.info("JSON storage initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize JSON storage: {str(e)}")
    
    def _init_csv(self):
        """Initialize CSV storage."""
        try:
            csv_path = str(self.db_path).replace('.json', '.csv')
            if not Path(csv_path).exists():
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['model_id', 'name', 'version', 'task_type', 'timestamp', 'model_path'])
            
            logger.info("CSV storage initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize CSV storage: {str(e)}")
    
    def register_model(
        self,
        name: str,
        model: Any,
        task_type: str,
        metrics: Dict = None,
        parameters: Dict = None,
        auto_version: bool = True
    ) -> str:
        """
        Register a trained model in the registry.
        
        Parameters
        ----------
        name : str
            Model name
        model : Any
            Trained model object
        task_type : str
            'classification' or 'regression'
        metrics : dict, optional
            Model metrics
        parameters : dict, optional
            Model hyperparameters
        auto_version : bool
            Auto-increment version
            
        Returns
        -------
        str
            Model ID/version
        """
        try:
            version = 1
            if auto_version:
                version = self._get_next_version(name)
            
            if self.storage_type == 'sqlite':
                return self._register_sqlite(name, model, task_type, metrics, parameters, version)
            elif self.storage_type == 'json':
                return self._register_json(name, model, task_type, metrics, parameters, version)
            elif self.storage_type == 'csv':
                return self._register_csv(name, model, task_type, metrics, parameters, version)
        
        except Exception as e:
            logger.error(f"Failed to register model: {str(e)}")
            return None
    
    def _register_sqlite(
        self, name: str, model: Any, task_type: str, metrics: Dict, parameters: Dict, version: int
    ) -> str:
        """Register model in SQLite."""
        try:
            # Save model to file in trained_models/
            model_dir = Path('trained_models')
            model_dir.mkdir(exist_ok=True)
            model_path = str(model_dir / f"{name}_v{version}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            # Store metadata in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO models (name, version, model_path, task_type, metrics, parameters)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                name,
                version,
                model_path,
                task_type,
                json.dumps(metrics or {}),
                json.dumps(parameters or {})
            ))
            conn.commit()
            conn.close()
            logger.info(f"Model {name} v{version} registered")
            return f"{name}_v{version}"
        except Exception as e:
            logger.error(f"SQLite registration failed: {str(e)}")
            return None
    
    def _register_json(
        self, name: str, model: Any, task_type: str, metrics: Dict, parameters: Dict, version: int
    ) -> str:
        """Register model in JSON."""
        try:
            # Save model to file in trained_models/
            model_dir = Path('trained_models')
            model_dir.mkdir(exist_ok=True)
            model_path = str(model_dir / f"{name}_v{version}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            # Load registry
            with open(self.db_path, 'r') as f:
                registry = json.load(f)
            # Add model entry
            model_id = f"{name}_v{version}"
            registry[model_id] = {
                'name': name,
                'version': version,
                'model_path': model_path,
                'task_type': task_type,
                'metrics': metrics or {},
                'parameters': parameters or {},
                'timestamp': datetime.now().isoformat()
            }
            # Save registry
            with open(self.db_path, 'w') as f:
                json.dump(registry, f, indent=2)
            logger.info(f"Model {name} v{version} registered")
            return model_id
        except Exception as e:
            logger.error(f"JSON registration failed: {str(e)}")
            return None
    
    def _register_csv(
        self, name: str, model: Any, task_type: str, metrics: Dict, parameters: Dict, version: int
    ) -> str:
        """Register model in CSV."""
        try:
            # Save model to file in trained_models/
            model_dir = Path('trained_models')
            model_dir.mkdir(exist_ok=True)
            model_path = str(model_dir / f"{name}_v{version}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            csv_path = str(self.db_path).replace('.json', '.csv')
            model_id = f"{name}_v{version}"
            # Append to CSV
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    model_id,
                    name,
                    version,
                    task_type,
                    datetime.now().isoformat(),
                    model_path
                ])
            logger.info(f"Model {name} v{version} registered in CSV")
            return model_id
        except Exception as e:
            logger.error(f"CSV registration failed: {str(e)}")
            return None
    
    def load_model(self, name: str, version: Optional[int] = None) -> Optional[Any]:
        """
        Load a model from registry.
        
        Parameters
        ----------
        name : str
            Model name
        version : int, optional
            Model version. Loads latest if None
            
        Returns
        -------
        model or None
            Loaded model
        """
        try:
            if version is None:
                version = self._get_latest_version(name)
            
            model_path = f"{name}_v{version}.pkl"
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"Loaded model {name} v{version}")
            return model
        
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return None
    
    def _get_next_version(self, name: str) -> int:
        """Get next version number for model."""
        try:
            if self.storage_type == 'sqlite':
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT MAX(version) FROM models WHERE name = ?', (name,))
                result = cursor.fetchone()
                conn.close()
                
                max_version = result[0] if result[0] is not None else 0
                return max_version + 1
            
            elif self.storage_type == 'json':
                with open(self.db_path, 'r') as f:
                    registry = json.load(f)
                
                versions = [
                    int(v.split('_v')[1]) for v in registry.keys()
                    if v.startswith(f"{name}_v")
                ]
                
                return max(versions) + 1 if versions else 1
            
            elif self.storage_type == 'csv':
                csv_path = str(self.db_path).replace('.json', '.csv')
                versions = []
                with open(csv_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row['name'] == name:
                            versions.append(int(row['version']))
                
                return max(versions) + 1 if versions else 1
        
        except:
            return 1
    
    def _get_latest_version(self, name: str) -> int:
        """Get latest version of model."""
        try:
            if self.storage_type == 'sqlite':
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT MAX(version) FROM models WHERE name = ?', (name,))
                result = cursor.fetchone()
                conn.close()
                
                return result[0] if result[0] is not None else 1
            
            elif self.storage_type == 'json':
                with open(self.db_path, 'r') as f:
                    registry = json.load(f)
                
                versions = [
                    int(v.split('_v')[1]) for v in registry.keys()
                    if v.startswith(f"{name}_v")
                ]
                
                return max(versions) if versions else 1
            
            elif self.storage_type == 'csv':
                csv_path = str(self.db_path).replace('.json', '.csv')
                versions = []
                with open(csv_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row['name'] == name:
                            versions.append(int(row['version']))
                
                return max(versions) if versions else 1
        
        except:
            return 1
    
    def list_models(self) -> List[Dict]:
        """
        List all registered models.
        
        Returns
        -------
        list
            List of model metadata
        """
        try:
            if self.storage_type == 'sqlite':
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT name, version, task_type, timestamp FROM models ORDER BY timestamp DESC')
                results = cursor.fetchall()
                conn.close()
                
                return [
                    {
                        'name': r[0],
                        'version': r[1],
                        'task_type': r[2],
                        'timestamp': r[3]
                    }
                    for r in results
                ]
            
            elif self.storage_type == 'json':
                with open(self.db_path, 'r') as f:
                    registry = json.load(f)
                
                return [
                    {
                        'model_id': k,
                        'name': v['name'],
                        'version': v['version'],
                        'task_type': v['task_type'],
                        'timestamp': v['timestamp']
                    }
                    for k, v in registry.items()
                ]
            
            elif self.storage_type == 'csv':
                csv_path = str(self.db_path).replace('.json', '.csv')
                models = []
                with open(csv_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        models.append({
                            'model_id': row['model_id'],
                            'name': row['name'],
                            'version': int(row['version']),
                            'task_type': row['task_type'],
                            'timestamp': row['timestamp']
                        })
                return models
        
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            return []
    
    def delete_model(self, name: str, version: int) -> bool:
        """
        Delete a model from registry.
        
        Parameters
        ----------
        name : str
            Model name
        version : int
            Model version
            
        Returns
        -------
        bool
            Success status
        """
        try:
            model_path = f"{name}_v{version}.pkl"
            
            if Path(model_path).exists():
                Path(model_path).unlink()
            
            if self.storage_type == 'sqlite':
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('DELETE FROM models WHERE name = ? AND version = ?', (name, version))
                conn.commit()
                conn.close()
            
            elif self.storage_type == 'json':
                with open(self.db_path, 'r') as f:
                    registry = json.load(f)
                
                model_id = f"{name}_v{version}"
                if model_id in registry:
                    del registry[model_id]
                
                with open(self.db_path, 'w') as f:
                    json.dump(registry, f, indent=2)
            
            elif self.storage_type == 'csv':
                csv_path = str(self.db_path).replace('.json', '.csv')
                models = []
                with open(csv_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if not (row['name'] == name and int(row['version']) == version):
                            models.append(row)
                
                with open(csv_path, 'w', newline='') as f:
                    if models:
                        writer = csv.DictWriter(f, fieldnames=['model_id', 'name', 'version', 'task_type', 'timestamp', 'model_path'])
                        writer.writeheader()
                        writer.writerows(models)
            
            logger.info(f"Deleted model {name} v{version}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete model: {str(e)}")
            return False
