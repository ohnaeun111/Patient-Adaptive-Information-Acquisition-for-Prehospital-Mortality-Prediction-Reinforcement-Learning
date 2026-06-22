import joblib
import numpy as np

class FrozenOracle:
    def __init__(self, pkl_path: str):
        models = joblib.load(pkl_path)
        if not isinstance(models, (list, tuple)) or len(models) < 1:
            raise ValueError("Unexpected pkl format. Expected tuple/list of classifiers.")
        self.models = list(models)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = []
        for m in self.models:
            p = m.predict_proba(X)[:, 1]
            probs.append(p)
        return np.mean(np.vstack(probs), axis=0)
