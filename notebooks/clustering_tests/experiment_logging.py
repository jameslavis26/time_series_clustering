import os
import json
import numpy as np
from datetime import datetime

class Experiment:
    def __init__(self, name, base_dir="experiments"):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.name = name
        self.timestamp = timestamp
        self.root = os.path.join(base_dir, name, timestamp)

        self.paths = {
            "results": os.path.join(self.root, "results"),
            "data": os.path.join(self.root, "data"),
            "models": os.path.join(self.root, "models"),
            "configs": os.path.join(self.root, "configs"),
            "logs": os.path.join(self.root, "logs"),
        }

        for p in self.paths.values():
            os.makedirs(p, exist_ok=True)

        # config
        self.config_path = os.path.join(self.paths["configs"], "config.json")
        self.config = {}
        self._save_config()

        # results dictionary
        self.results = {}
        self.results_path = os.path.join(self.paths["results"], "results.json")

    # -------------------------
    # Config handling
    # -------------------------
    def add_config(self, **kwargs):
        self.config.update(kwargs)
        self._save_config()

    def _save_config(self):
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=2)

    # -------------------------
    # Result handling
    # -------------------------
    def add_result(self, **kwargs):
        self.results.update(kwargs)
        self._save_results()

    def add_results_bulk(self, results_dict):
        self.results.update(results_dict)
        self._save_results()

    def _save_results(self):
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(self.results_path, "w") as f:
            json.dump({k: convert(v) for k, v in self.results.items()}, f, indent=2)

    def get_result(self, key):
        return self.results.get(key)

    # -------------------------
    # Dataset handling with metadata
    # -------------------------
    def save_dataset(self, data, name, metadata=None):
        """
        Save a dataset with optional metadata.
        - data: np.ndarray or compatible
        - name: filename without extension
        - metadata: dict
        Saves:
            data -> data/{name}.npy
            metadata -> data/{name}_metadata.json
        """
        # save array
        data_path = os.path.join(self.paths["data"], f"{name}.npy")
        np.save(data_path, data)

        # save metadata if provided
        if metadata is not None:
            meta_path = os.path.join(self.paths["data"], f"{name}_metadata.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

    # -------------------------
    # Other saving helpers
    # -------------------------
    def save_numpy(self, array, name):
        path = os.path.join(self.paths["results"], f"{name}.npy")
        np.save(path, array)

    def save_scores(self, scores_dict, name="scores"):
        path = os.path.join(self.paths["results"], f"{name}.json")
        with open(path, "w") as f:
            json.dump(scores_dict, f, indent=2)

    def save_model(self, model, name):
        path = os.path.join(self.paths["models"], f"{name}.pkl")
        with open(path, "wb") as f:
            import pickle
            pickle.dump(model, f)

    # -------------------------
    # Logging
    # -------------------------
    def log(self, msg):
        log_path = os.path.join(self.paths["logs"], "log.txt")
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"

        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")
