"""Stage 4 (V1): baseline LightGBM matcher and its validation scores.

Trains a LightGBM binary classifier on data/marts/matcher/matcher_train_pairs
.parquet using the 16 pairwise features (pair_features.FEATURE_COLUMNS) with
the natural class distribution and LightGBM's default parameters (no
resampling, no tuning, no early stopping). Scores every pair of
matcher_val_pairs.parquet and reports:

- PR-AUC (average precision) and ROC-AUC
- pair-level precision / recall / F0.5 / predicted positives over a
  threshold sweep, plus the exact F0.5-optimal threshold
- the challenge metric (macro F0.5 per S1, singletons included, blocker
  misses counted as false negatives) at the same thresholds, with no S1-level
  post-processing
- feature importance (gain and split count)

Writes to data/models/: the model (matcher_v1_lgbm.txt), validation
probabilities (matcher_v1_val_predictions.parquet), the sweep table and a
JSON report. Test data is not touched.

``--version`` picks the feature set (see VERSIONS): v2 / v2_nocount train
the same model (same parameters) on the V2 datasets, v3a additionally joins
the row-aligned V3a add-on files; outputs are written as <model_name>* files.
Train-set metrics are computed on every 4th training row.

Run from code/business_entity_resolution/:
    python3 src/pipeline/run_stage4_train_matcher_v1.py [--version v2|v2_nocount|v3a]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from config import MATCHER_DIR, MODELS_DIR, SEED, STAGING_DIR  # noqa: E402
from metrics import per_entity_fbeta  # noqa: E402
from pair_features import FEATURE_COLUMNS  # noqa: E402
from pair_features_v2 import V2_FEATURE_COLUMNS  # noqa: E402
from pair_features_v3a import V3A_COLUMNS  # noqa: E402
from split import load_matcher_split  # noqa: E402

# V1 features with zero gain in the V1 model: country is constant (blocking
# is country-scoped) and the exact-match flags are implied by Jaccard == 1.
DROPPED_IN_V2 = {"country_exact_match", "address_exact_match", "name_and_address_exact_match"}
V2_MODEL_FEATURES = [c for c in V2_FEATURE_COLUMNS if c not in DROPPED_IN_V2]
V2_NOCOUNT_FEATURES = [c for c in V2_MODEL_FEATURES if c != "s1_candidate_count"]
# each version: the model name and its feature sources, as (dataset suffix,
# columns) pairs; several sources are row-aligned files joined by position
VERSIONS = {
    "v1": {"model_name": "matcher_v1_lgbm", "sources": [("", FEATURE_COLUMNS)]},
    "v2": {"model_name": "matcher_v2_lgbm", "sources": [("_v2", V2_MODEL_FEATURES)]},
    # ablation: V2 without the pool-size-dependent candidate-count prior
    "v2_nocount": {"model_name": "matcher_v2_nocount_lgbm",
                   "sources": [("_v2", V2_NOCOUNT_FEATURES)]},
    # v2_nocount + the V3a add-on features (India error modes)
    "v3a": {"model_name": "matcher_v3a_lgbm",
            "sources": [("_v2", V2_NOCOUNT_FEATURES), ("_v3a", V3A_COLUMNS)]},
}
# train-set metrics are computed on every TRAIN_METRIC_STEP-th row, so the full
# training matrix never has to be held next to LightGBM's binned dataset
TRAIN_METRIC_STEP = 4
PREDICT_CHUNK = 5_000_000
# LightGBM defaults, made explicit; only seeds / determinism added.
PARAMS = {
    "objective": "binary",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}
NUM_BOOST_ROUND = 100
THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95,
              0.96, 0.97, 0.98, 0.99]
BETA = 0.5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_xy(sources, extra_columns=(), row_step=1):
    """Feature matrix, target and optional extra columns.

    ``sources`` is a list of (path, columns) of row-aligned pairs files whose
    columns are placed side by side; their targets must agree row for row.
    With ``row_step`` > 1 only rows whose global index is a multiple of it
    are kept. The float32 matrix is filled batch by batch so a full Arrow
    table is never held alongside it.
    """
    n = pq.ParquetFile(sources[0][0]).metadata.num_rows
    keep = -(-n // row_step)
    x = np.empty((keep, sum(len(cols) for _, cols in sources)), dtype=np.float32)
    y, col0 = None, 0
    for path, cols in sources:
        pf = pq.ParquetFile(path)
        if pf.metadata.num_rows != n:
            raise ValueError(f"{path} has {pf.metadata.num_rows} rows, expected {n}")
        ys, start = np.empty(keep, dtype=np.int8), 0
        for batch in pf.iter_batches(batch_size=2_000_000, columns=[*cols, "target"]):
            sel = slice((-start) % row_step, None, row_step)
            out = -(-start // row_step)
            target = batch["target"].to_numpy()[sel]
            for j, col in enumerate(cols):
                x[out:out + len(target), col0 + j] = batch[col].to_numpy()[sel]
            ys[out:out + len(target)] = target
            start += batch.num_rows
        if y is None:
            y = ys
        elif not np.array_equal(y, ys):
            raise ValueError(f"{path} is not row-aligned with {sources[0][0]}")
        col0 += len(cols)
    extra = pq.read_table(sources[0][0], columns=list(extra_columns)) if extra_columns else None
    return x, y, extra


def predict_chunked(model, x):
    """Predict in row slices: LightGBM may copy its input to float64, which
    for the full ~41M x 35 matrix alone is ~11 GB."""
    return np.concatenate([model.predict(x[i:i + PREDICT_CHUNK])
                           for i in range(0, len(x), PREDICT_CHUNK)])


def fbeta(p, r, beta=BETA):
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) > 0 else 0.0


def pair_metrics(y, prob, t):
    pred = prob >= t
    tp = int((pred & (y == 1)).sum())
    n_pred = int(pred.sum())
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / int(y.sum())
    return precision, recall, fbeta(precision, recall), n_pred


def best_pair_threshold(y, prob):
    """Exact F0.5-optimal threshold over all distinct predicted probabilities."""
    order = np.argsort(-prob, kind="stable")
    p_sorted, tp = prob[order], np.cumsum(y[order], dtype=np.int64)
    k = np.arange(1, len(y) + 1)
    # evaluate only at the last index of each run of equal probabilities
    last = np.flatnonzero(np.r_[p_sorted[1:] != p_sorted[:-1], True])
    precision, recall = tp[last] / k[last], tp[last] / tp[-1]
    b2 = BETA * BETA
    f = np.where(b2 * precision + recall > 0,
                 (1 + b2) * precision * recall / (b2 * precision + recall), 0.0)
    i = int(np.argmax(f))
    return float(p_sorted[last[i]]), float(precision[i]), float(recall[i]), float(f[i])


class ChallengeScorer:
    """Macro F0.5 per S1 (the leaderboard metric) from pair predictions.

    Covers every matcher_val S1: ground-truth counts include true matches the
    blocker missed, and S1s with no candidates predict an empty list.
    """

    def __init__(self, s1_ids: pa.ChunkedArray):
        split = load_matcher_split()
        val_ids = split.loc[split["split"] == "matcher_val", "entity_id"].to_numpy()
        gt = pq.read_table(STAGING_DIR / "train" / "ground_truth.parquet").to_pandas()
        gt = gt.set_index("source1_entity_id").reindex(val_ids)["matched_entity_ids"]
        n_true = gt.fillna("").map(lambda s: len(s.split(",")) if s else 0)
        self.n_true = n_true.to_numpy()
        enc = pc.dictionary_encode(s1_ids.combine_chunks())
        dict_pos = pd.Index(val_ids).get_indexer(enc.dictionary.to_numpy(zero_copy_only=False))
        if (dict_pos < 0).any():
            raise ValueError("validation pairs reference S1 IDs outside matcher_val")
        self.code, self.n = dict_pos[enc.indices.to_numpy()], len(val_ids)

    def score(self, pred: np.ndarray, y: np.ndarray) -> float:
        n_pred = np.bincount(self.code[pred], minlength=self.n)
        n_correct = np.bincount(self.code[pred & (y == 1)], minlength=self.n)
        return float(per_entity_fbeta(n_pred, n_correct, self.n_true, BETA).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=sorted(VERSIONS), default="v1")
    version = VERSIONS[ap.parse_args().version]
    model_name = version["model_name"]
    features = [c for _, cols in version["sources"] for c in cols]

    def sources(split):
        return [(MATCHER_DIR / f"matcher_{split}_pairs{suffix}.parquet", cols)
                for suffix, cols in version["sources"]]

    t_start = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ train
    x, y, _ = load_xy(sources("train"))
    log(f"train: {len(y):,} pairs, {int(y.sum()):,} positive ({y.mean():.4%}), "
        f"{len(features)} features")
    train_set = lgb.Dataset(x, label=y, feature_name=features, free_raw_data=True)
    train_set.construct()
    del x, y  # LightGBM keeps its own binned copy
    t0 = time.time()
    model = lgb.train(PARAMS, train_set, num_boost_round=NUM_BOOST_ROUND)
    log(f"trained {NUM_BOOST_ROUND} rounds in {time.time() - t0:.1f}s")
    del train_set
    xs, ys, _ = load_xy(sources("train"), row_step=TRAIN_METRIC_STEP)
    train_prob = predict_chunked(model, xs)
    train_scores = {
        "average_precision": float(average_precision_score(ys, train_prob)),
        "roc_auc": float(roc_auc_score(ys, train_prob)),
        "rows_scored": int(len(ys)),
    }
    del xs, ys, train_prob
    model_path = MODELS_DIR / f"{model_name}.txt"
    model.save_model(str(model_path))

    # ------------------------------------------------------- validation
    xv, yv, ids = load_xy(
        sources("val"), extra_columns=("source1_entity_id", "candidate_entity_id"),
    )
    t0 = time.time()
    prob = predict_chunked(model, xv)
    log(f"scored {len(yv):,} validation pairs in {time.time() - t0:.1f}s")
    del xv
    pred_path = MODELS_DIR / f"{model_name}_val_predictions.parquet"
    pq.write_table(
        ids.append_column("target", pa.array(yv))
           .append_column("pred_prob", pa.array(prob.astype(np.float32))),
        pred_path, row_group_size=2_000_000,
    )

    ap = float(average_precision_score(yv, prob))
    roc = float(roc_auc_score(yv, prob))
    scorer = ChallengeScorer(ids["source1_entity_id"])
    del ids

    sweep = []
    for t in THRESHOLDS:
        precision, recall, f, n_pred = pair_metrics(yv, prob, t)
        sweep.append({
            "threshold": t, "precision": precision, "recall": recall, "f0_5": f,
            "predicted_positives": n_pred,
            "challenge_macro_f0_5": scorer.score(prob >= t, yv),
        })
    sweep = pd.DataFrame(sweep)
    best_sweep = sweep.loc[sweep["f0_5"].idxmax()]
    t_opt, p_opt, r_opt, f_opt = best_pair_threshold(yv, prob)
    macro_opt = scorer.score(prob >= t_opt, yv)
    sweep.to_csv(MODELS_DIR / f"{model_name}_threshold_sweep.tsv", sep="\t", index=False)

    gain = model.feature_importance("gain")
    importance = pd.DataFrame({
        "feature": features,
        "gain": gain,
        "gain_share": gain / gain.sum(),
        "split_count": model.feature_importance("split"),
    }).sort_values("gain", ascending=False)

    report = {
        "model": str(model_path),
        "params": PARAMS,
        "num_boost_round": NUM_BOOST_ROUND,
        "features": features,
        "validation": {
            "candidate_pairs": int(len(yv)),
            "positive_pairs": int(yv.sum()),
            "positive_rate": float(yv.mean()),
            "average_precision": ap,
            "roc_auc": roc,
            "best_sweep_threshold": float(best_sweep["threshold"]),
            "best_sweep_f0_5": float(best_sweep["f0_5"]),
            "optimal_threshold": t_opt,
            "optimal_precision": p_opt,
            "optimal_recall": r_opt,
            "optimal_f0_5": f_opt,
            "challenge_macro_f0_5_at_optimal": macro_opt,
        },
        "train": train_scores,
        "threshold_sweep": sweep.to_dict(orient="records"),
        "feature_importance": importance.to_dict(orient="records"),
        "predictions": str(pred_path),
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    with open(MODELS_DIR / f"{model_name}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    v = report["validation"]
    print(f"\nvalidation pairs: {v['candidate_pairs']:,} | positives: {v['positive_pairs']:,} "
          f"| positive rate: {v['positive_rate']:.4%}")
    print(f"PR-AUC (AP): {ap:.4f} | ROC-AUC: {roc:.4f} "
          f"(train AP {train_scores['average_precision']:.4f}, ROC {train_scores['roc_auc']:.4f})")
    print("\n" + sweep.to_string(index=False, float_format=lambda z: f"{z:.4f}"))
    print(f"\nbest sweep threshold: {best_sweep['threshold']:.2f} -> F0.5 {best_sweep['f0_5']:.4f}")
    print(f"exact optimum: threshold {t_opt:.4f} -> P {p_opt:.4f} R {r_opt:.4f} "
          f"F0.5 {f_opt:.4f} | challenge macro F0.5 {macro_opt:.4f}")
    print("\nfeature importance:\n" + importance.to_string(
        index=False, float_format=lambda z: f"{z:,.4f}"))
    log(f"done in {time.time() - t_start:.0f}s -> {MODELS_DIR}")


if __name__ == "__main__":
    main()
