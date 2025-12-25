from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


class Metrics:
    """Simple metrics utilities for evaluation."""

    @staticmethod
    def pearson_correlation(xs: List[float], ys: List[float]) -> float:
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        if xs.size == 0:
            return float('nan')
        if xs.std() == 0 or ys.std() == 0:
            return 0.0
        return float(np.corrcoef(xs, ys)[0, 1])

    @staticmethod
    def precision_recall(y_true: List[int], y_pred: List[int], average: str = 'binary') -> Dict[str, float]:
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
        return {"precision": float(p), "recall": float(r), "f1": float(f1)}

    @staticmethod
    def precision_recall_per_class(y_true: List[int], y_pred: List[int], labels: List[int]) -> Dict[int, Dict[str, float]]:
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        out = {}
        for i, lab in enumerate(labels):
            out[int(lab)] = {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i])}
        return out
