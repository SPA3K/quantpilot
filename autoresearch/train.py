#!/usr/bin/env python3
"""
训练脚本 — Agent 可自由修改此文件。
当前: LightGBM baseline。
目标: 最大化 test 集 conf_acc。
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, pickle, datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import load_data, evaluate

MODEL_DIR = Path.home() / "workspace" / "quantpilot" / "models"

def train_and_eval():
    X_train, y_train, X_val, y_val, X_test, y_test, features, meta = load_data()

    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        num_leaves=15, min_child_samples=15,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=0.5, verbose=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(30, verbose=False)])

    val_metrics = evaluate(model, {**meta, "X_test": X_val, "y_test": y_val, "fwd_test": np.zeros(len(y_val))})
    test_metrics = evaluate(model, meta)

    print(f"val_acc={val_metrics['acc']:.2%}  val_conf_acc={val_metrics['conf_acc']:.2%} ({val_metrics['conf_n']}样本)")
    print(f"test_acc={test_metrics['acc']:.2%}  test_conf_acc={test_metrics['conf_acc']:.2%} ({test_metrics['conf_n']}样本)")
    print(f"test_conf_avg_fwd_ret={test_metrics['conf_avg_fwd_ret']:+.4%}")
    return model, features, val_metrics, test_metrics

def log_experiment(val_m, test_m, note=""):
    log = Path(__file__).parent / "experiments.log"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    line = f"{now} | {note} | val={val_m['conf_acc']:.2%} | test_conf={test_m['conf_acc']:.2%}({test_m['conf_n']}) | fwd={test_m['conf_avg_fwd_ret']:+.3%}\n"
    with open(log, "a", encoding="utf-8") as f:
        f.write(line)
    print(f"logged: {line.strip()}")

if __name__ == "__main__":
    model, features, val_m, test_m = train_and_eval()
    code = os.environ.get("AUTORESEARCH_CODE", "single")
    pkl = MODEL_DIR / f"autoresearch_{code}.pkl"
    pkl.write_bytes(pickle.dumps(model))
    log_experiment(val_m, test_m, "baseline LightGBM")
