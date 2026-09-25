"""Challenge metric: macro-averaged F_beta over Source 1 entities.

Per S1 entity, with P = |pred & true| / |pred| and R = |pred & true| / |true|:

    F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)

averaged over every S1 entity, singletons included. A singleton scores 1.0
when the predicted list is empty and 0.0 otherwise; a non-singleton with an
empty prediction (or no correct IDs) scores 0.0.
"""

import numpy as np


def per_entity_fbeta(n_pred, n_correct, n_true, beta: float = 0.5) -> np.ndarray:
    """Vectorized per-entity F_beta from counts.

    ``n_pred`` = size of the predicted list, ``n_correct`` = how many of
    those are true matches, ``n_true`` = size of the ground-truth list.
    """
    n_pred = np.asarray(n_pred, dtype=np.float64)
    n_correct = np.asarray(n_correct, dtype=np.float64)
    n_true = np.asarray(n_true, dtype=np.float64)
    b2 = beta * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n_pred > 0, n_correct / n_pred, 0.0)
        r = np.where(n_true > 0, n_correct / n_true, 0.0)
        denom = b2 * p + r
        f = np.where(denom > 0, (1 + b2) * p * r / denom, 0.0)
    singleton = n_true == 0
    return np.where(singleton, (n_pred == 0).astype(np.float64), f)


def macro_fbeta(n_pred, n_correct, n_true, beta: float = 0.5) -> float:
    return float(per_entity_fbeta(n_pred, n_correct, n_true, beta).mean())
