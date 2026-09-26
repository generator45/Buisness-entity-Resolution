"""Stage 4b: LightGBM capacity tuning with early stopping on a train-side holdout.

Uses one feature set from stage 4's VERSIONS (default v3c). The matcher_train
pairs are split at an S1 boundary into a fit part (first ~90% of rows) and an
early-stopping holdout (last ~10%). Rows are grouped by S1 in source-file
order, which is effectively random (checked: the tail matches the whole on
positive rate, India share and candidates per S1), so the holdout is a random
set of S1 entities. matcher_val is never used for any choice here.

Every config in CONFIGS trains on the fit part with early stopping on holdout
log-loss (up to MAX_ROUNDS), and is scored by holdout average precision. The
best config's model is then evaluated on matcher_val exactly like stage 4
(evaluate_on_val) and saved as matcher_<version>_tuned_lgbm*.

Memory: the training matrix is a memory-mapped .npy file (file-backed pages)
and the fit / holdout datasets are built from contiguous slices of it, so
the raw matrix is never copied into RAM. (LightGBM's Dataset.subset was
avoided: it turns the row indices into a Python list and copies the bins,
which ran out of memory.)

Run from code/business_entity_resolution/:
    python3 src/pipeline/run_stage4b_tune_matcher.py [--version v3c] [--configs leaves255]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from config import MATCHER_DIR, MODELS_DIR, SEED  # noqa: E402
from run_stage4_train_matcher_v1 import (  # noqa: E402
    MATRIX_DIR,
    PARAMS,
    VERSIONS,
    evaluate_on_val,
    load_xy,
    log,
    predict_chunked,
)

HOLDOUT_FRACTION = 0.10
MAX_ROUNDS = 600
EARLY_STOPPING_ROUNDS = 30
_BASE = {k: v for k, v in PARAMS.items() if k not in ("learning_rate", "num_leaves")}
_BASE["metric"] = "binary_logloss"
_REGULARIZED = {"feature_fraction": 0.8, "bagging_fraction": 0.5, "bagging_freq": 1,
                "lambda_l2": 1.0, "learning_rate": 0.1}
CONFIGS = {
    # the stage-4 defaults, but trained to convergence: separates "more
    # rounds" from "bigger trees"
    "default_early_stopped": {**_BASE, "learning_rate": 0.1, "num_leaves": 31},
    "leaves127": {**_BASE, **_REGULARIZED, "num_leaves": 127, "min_data_in_leaf": 200},
    "leaves255": {**_BASE, **_REGULARIZED, "num_leaves": 255, "min_data_in_leaf": 500},
}


def holdout_cut(path):
    """First row of the holdout: the first S1 group starting at >= 90% of rows."""
    enc = pc.dictionary_encode(
        pq.read_table(path, columns=["source1_entity_id"])["source1_entity_id"].combine_chunks())
    codes = enc.indices.to_numpy()
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    return int(starts[np.searchsorted(starts, int((1 - HOLDOUT_FRACTION) * len(codes)))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=sorted(VERSIONS), default="v3c")
    ap.add_argument("--configs", nargs="+", choices=sorted(CONFIGS), default=sorted(CONFIGS),
                    help="subset of CONFIGS to train (default: all)")
    args = ap.parse_args()
    version_name = args.version
    configs = {k: CONFIGS[k] for k in args.configs}
    version = VERSIONS[version_name]
    features = [c for _, cols in version["sources"] for c in cols]
    model_name = f"matcher_{version_name}_tuned_lgbm"

    def sources(split):
        return [(MATCHER_DIR / f"matcher_{split}_pairs{suffix}.parquet", cols)
                for suffix, cols in version["sources"]]

    t_start = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_matrix = MATRIX_DIR / f"{model_name}_train_x.npy"
    x, y, _ = load_xy(sources("train"), memmap_path=train_matrix)
    cut = holdout_cut(sources("train")[0][0])
    y_hold = y[cut:]
    log(f"train: {len(y):,} pairs, {len(features)} features | fit {cut:,} rows, "
        f"holdout {len(y_hold):,} rows ({y_hold.mean():.4%} positive)")
    # contiguous slices of the memmap: views, no copies
    fit_set = lgb.Dataset(x[:cut], label=y[:cut], feature_name=features, free_raw_data=True,
                          params={"verbosity": -1, "seed": SEED})
    hold_set = lgb.Dataset(x[cut:], label=y_hold, reference=fit_set, free_raw_data=True)
    fit_set.construct()
    hold_set.construct()
    x_hold = x[cut:]
    log("binned datasets built")

    results, models = [], {}
    for name, params in configs.items():
        t0 = time.time()
        model = lgb.train(
            params, fit_set, num_boost_round=MAX_ROUNDS,
            valid_sets=[hold_set], valid_names=["holdout"],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                       lgb.log_evaluation(100)],
        )
        prob = predict_chunked(model, x_hold)
        results.append({
            "config": name, "best_iteration": model.best_iteration,
            "holdout_logloss": model.best_score["holdout"]["binary_logloss"],
            "holdout_average_precision": float(average_precision_score(y_hold, prob)),
            "holdout_roc_auc": float(roc_auc_score(y_hold, prob)),
            "train_seconds": round(time.time() - t0, 1),
        })
        models[name] = model
        log(f"{name}: {results[-1]}")
    table = pd.DataFrame(results).sort_values("holdout_average_precision", ascending=False)
    best = table.iloc[0]["config"]
    print("\ntuning (train-side holdout; matcher_val not used):\n"
          + table.to_string(index=False, float_format=lambda z: f"{z:.5f}"))
    log(f"selected: {best}")
    del fit_set, hold_set, x_hold, x
    train_matrix.unlink()

    model = models[best]
    model_path = MODELS_DIR / f"{model_name}.txt"
    model.save_model(str(model_path), num_iteration=model.best_iteration)
    hold_row = table.set_index("config").loc[best]
    report = evaluate_on_val(model, model_name, features, sources("val"), {
        "model": str(model_path),
        "params": CONFIGS[best],
        "num_boost_round": int(model.best_iteration),
        "selected_config": best,
        "tuning": table.to_dict(orient="records"),
        # train-side scores are on the early-stopping holdout here
        "train": {"average_precision": float(hold_row["holdout_average_precision"]),
                  "roc_auc": float(hold_row["holdout_roc_auc"]),
                  "rows_scored": int(len(y_hold)), "scope": "train-side holdout"},
    }, t_start)
    with open(MODELS_DIR / f"{model_name}_tuning.json", "w") as f:
        json.dump({"selected": best, "configs": CONFIGS, "results": results,
                   "validation_challenge_macro_f0_5_at_optimal":
                       report["validation"]["challenge_macro_f0_5_at_optimal"]}, f, indent=2)


if __name__ == "__main__":
    main()
