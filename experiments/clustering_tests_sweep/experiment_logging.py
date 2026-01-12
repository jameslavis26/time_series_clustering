import os
import json
import numpy as np
from datetime import datetime


class Experiment:
    def __init__(self, name, base_dir="experiments", resume=False):
        self.name = name

        if resume:
            # resume latest experiment
            root = os.path.join(base_dir, name)
            timestamps = sorted(os.listdir(root))
            self.timestamp = timestamps[-1]
        else:
            self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.root = os.path.join(base_dir, name, self.timestamp)

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
        self.config = self._load_json(self.config_path, default={})
        self._save_config()

        # results
        self.results_path = os.path.join(self.paths["results"], "results.json")
        self.results = self._load_json(self.results_path, default={})

    # -------------------------
    # Internal helpers
    # -------------------------
    def _json_safe(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._json_safe(v) for v in obj]
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)

    def _atomic_write(self, path, data):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    def _load_json(self, path, default=None):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return default

    # -------------------------
    # Config handling
    # -------------------------
    def add_config(self, **kwargs):
        self.config.update(kwargs)
        self._save_config()

    def _save_config(self):
        self._atomic_write(self.config_path, self._json_safe(self.config))

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
        self._atomic_write(self.results_path, self._json_safe(self.results))

    def get_result(self, key):
        return self.results.get(key)

    # -------------------------
    # Dataset handling
    # -------------------------
    def save_dataset(self, data, name, metadata=None):
        data_path = os.path.join(self.paths["data"], f"{name}.npy")
        np.save(data_path, data)

        if metadata is not None:
            meta_path = os.path.join(self.paths["data"], f"{name}_metadata.json")
            self._atomic_write(meta_path, self._json_safe(metadata))

    def load_dataset(self, name, with_metadata=False):
        data_path = os.path.join(self.paths["data"], f"{name}.npy")
        data = np.load(data_path)

        if not with_metadata:
            return data

        meta_path = os.path.join(self.paths["data"], f"{name}_metadata.json")
        metadata = self._load_json(meta_path, default=None)

        return data, metadata

    # -------------------------
    # Other saving helpers
    # -------------------------
    def save_numpy(self, array, name):
        path = os.path.join(self.paths["results"], f"{name}.npy")
        np.save(path, array)

    def save_dataframe(self, df, name):
        path = os.path.join(self.paths["results"], f"{name}.csv")
        df.to_csv(path, index=False)

    def save_scores(self, scores_dict, name="scores"):
        path = os.path.join(self.paths["results"], f"{name}.json")
        self._atomic_write(path, self._json_safe(scores_dict))

    def save_model(self, model, name):
        import pickle
        path = os.path.join(self.paths["models"], f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)

    # -------------------------
    # Logging
    # -------------------------
    def log(self, msg):
        log_path = os.path.join(self.paths["logs"], "log.txt")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] [{self.name}] {msg}"

        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")
