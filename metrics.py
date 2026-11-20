import numpy as np


def scores(y, probabilities, thresholds):
    """Labels are 1/0; -1 (uncertain pseudo label) is excluded from that class's counts."""
    y = np.asarray(y)
    p = np.asarray(probabilities) >= np.asarray(thresholds)
    if not len(y):
        return {"n_windows": 0}
    known = y >= 0
    y = y == 1
    tp = (y & p).sum(0)
    fp = (~y & p & known).sum(0)
    fn = (y & ~p).sum(0)
    precision = np.divide(tp, tp + fp, out=np.zeros(5, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros(5, dtype=float), where=(tp + fn) > 0)
    f1 = np.divide(2*tp, 2*tp + fp + fn, out=np.zeros(5, dtype=float), where=(2*tp + fp + fn) > 0)
    neg = ~y.any(1) & known.all(1)
    result = {"n_windows": len(y), "precision": precision.tolist(), "recall": recall.tolist(),
              "f1": f1.tolist(), "macro_f1": float(f1.mean()), "support": y.sum(0).tolist(),
              "tp": tp.tolist(), "fp": fp.tolist(), "fn": fn.tolist(),
              "negative_windows": int(neg.sum()),
              "negative_false_positive_windows": int(p[neg].any(1).sum()),
              "negative_window_false_positive_rate": float(p[neg].any(1).mean()) if neg.any() else None}
    if not known.all():
        result["uncertain_labels_ignored"] = int((~known).sum())
    return result


def tune_thresholds(y, probabilities):
    """Validation only. Use a declared fixed grid, never test labels. -1 labels are skipped."""
    y, probabilities = np.asarray(y), np.asarray(probabilities)
    thresholds = []
    for i in range(5):
        known = y[:, i] >= 0
        yy, pp = y[known, i] == 1, probabilities[known, i]
        if not yy.any() or yy.all():
            raise ValueError(f"Validation needs positive and negative windows for class {i}")
        best = (-1, -1)
        for t in np.arange(0.05, 1.0, 0.05):
            pred = pp >= t
            tp = int((pred & yy).sum())
            fp = int((pred & ~yy).sum())
            fn = int((~pred & yy).sum())
            f1 = 2*tp / max(1, 2*tp + fp + fn)
            if (f1, float(t)) > best:
                best = (f1, float(t))
        thresholds.append(best[1])
    return thresholds
