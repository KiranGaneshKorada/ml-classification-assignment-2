"""Phase 3 validation: saved models, metrics, train/test separation and reproducibility.

Usage: python3 -m src.validate_models   (after python3 -m src.train_models)
Metrics are recomputed here directly with sklearn, independently of evaluate_models.py.
Exits with a non-zero status if any check fails.
"""
import hashlib
import re
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluate_models import METRIC_COLUMNS
from src.preprocessing import (
    NUMERICAL_FEATURES, PROJECT_ROOT, RAW_DATA_PATH, TEST_DATA_PATH, clean_features,
    load_raw_data,
)
from src.train_models import (
    COMPARISON_PATH, CONFUSION_DIR, MODEL_DIR, MODEL_NAMES, PREDICTIONS_PATH,
    REPORTS_DIR, build_model_pipeline, build_classifiers, load_datasets,
)

ORIGINAL_SHA1 = "252c7d50740da7988d71174d15184247463d975c"
TEST_CSV_SHA1 = "356ad78f5bfeeab3ebdb22891e341995520695cd"  # recorded after Phase 2
EXPECTED_TYPES = {
    "logistic_regression": (LogisticRegression, {"max_iter": 2000, "random_state": 42}),
    "decision_tree": (DecisionTreeClassifier, {"random_state": 42}),
    "knn": (KNeighborsClassifier, {"n_neighbors": 5}),
    "naive_bayes": (GaussianNB, {}),
    "svm": (SVC, {"probability": True, "random_state": 42}),
}

results = []


def check(name, passed, detail=""):
    results.append(bool(passed))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def sha1(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()


def sklearn_metrics(y_true, y_pred, y_proba):
    kw = dict(pos_label=1, average="binary", zero_division=0)
    return {"Accuracy": accuracy_score(y_true, y_pred), "AUC": roc_auc_score(y_true, y_proba),
            "Precision": precision_score(y_true, y_pred, **kw),
            "Recall": recall_score(y_true, y_pred, **kw), "F1": f1_score(y_true, y_pred, **kw),
            "MCC": matthews_corrcoef(y_true, y_pred)}


def main():
    # Data integrity.
    check("Original CSV unchanged", sha1(RAW_DATA_PATH) == ORIGINAL_SHA1)
    check("Phase 2 test CSV unchanged", sha1(TEST_DATA_PATH) == TEST_CSV_SHA1)

    X_train, y_train, X_test, y_test = load_datasets()  # raises if test CSV != split
    raw = load_raw_data()
    train_ids = set(raw.loc[X_train.index, "customerID"])
    test_ids = set(raw.drop(index=X_train.index)["customerID"])
    check("Train and test customers are disjoint", not (train_ids & test_ids),
          f"train={len(train_ids)}, test={len(test_ids)}, shared={len(train_ids & test_ids)}")
    check("Saved test CSV equals the Phase 2 split's test rows", len(X_test) == len(test_ids))

    Xt_train = None
    comparison = pd.read_csv(COMPARISON_PATH, index_col="Model")
    stored = pd.read_csv(PREDICTIONS_PATH)
    check("Stored y_true equals test CSV labels", stored["y_true"].equals(y_test))
    check("model_comparison.csv has 5 models x 6 metrics",
          list(comparison.columns) == METRIC_COLUMNS
          and list(comparison.index) == list(MODEL_NAMES.values())
          and comparison.notna().all().all(), f"shape={comparison.shape}")

    train_means = clean_features(X_train)[NUMERICAL_FEATURES].mean().to_numpy()
    fresh_classifiers = build_classifiers()
    for key, name in MODEL_NAMES.items():
        print(f"\n--- {name} ---")
        path = MODEL_DIR / f"{key}.joblib"
        model = joblib.load(path)
        check("Loads with joblib as Pipeline(preprocessor -> classifier)",
              isinstance(model, Pipeline) and list(model.named_steps) == ["preprocessor", "classifier"],
              path.name)
        expected_type, expected_params = EXPECTED_TYPES[key]
        clf = model.named_steps["classifier"]
        params = clf.get_params()
        check("Classifier type and configuration as specified",
              isinstance(clf, expected_type)
              and all(params[p] == v for p, v in expected_params.items()), repr(clf))

        # Fitted on training rows only.
        preprocessor = model.named_steps["preprocessor"]
        scaler = preprocessor.named_steps["encode"].named_transformers_["num"]
        check("Preprocessor fitted on the 5,634 training rows only",
              scaler.n_samples_seen_ == len(X_train) and np.allclose(scaler.mean_, train_means))
        Xt_train = preprocessor.transform(X_train)
        if key == "decision_tree":
            detail = clf.tree_.n_node_samples[0]
            ok = detail == len(X_train)
        elif key == "knn":
            detail = clf.n_samples_fit_
            ok = detail == len(X_train) and np.array_equal(clf._fit_X, Xt_train)
        elif key == "naive_bayes":
            detail = clf.class_count_.tolist()
            ok = np.array_equal(clf.class_count_, np.bincount(y_train))
        elif key == "svm":
            detail = f"{len(clf.support_)} support vectors"
            ok = np.array_equal(clf.support_vectors_, Xt_train[clf.support_])
        else:
            detail = f"n_features_in_={clf.n_features_in_}"
            ok = clf.n_features_in_ == Xt_train.shape[1]
        check("Classifier fitted on training rows (model-specific evidence)", ok, detail)

        # Loaded model reproduces stored predictions and the saved metrics.
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
        check("Loaded model reproduces stored predictions",
              np.array_equal(y_pred, stored[f"{key}_pred"])
              and np.allclose(y_proba, stored[f"{key}_proba"], rtol=0, atol=1e-12))
        recomputed = sklearn_metrics(y_test, y_pred, y_proba)
        diffs = {m: abs(round(recomputed[m], 4) - comparison.loc[name, m]) for m in METRIC_COLUMNS}
        check("Six metrics recomputed from loaded model match model_comparison.csv",
              max(diffs.values()) < 1e-9,
              ", ".join(f"{m}={recomputed[m]:.4f}" for m in METRIC_COLUMNS))
        auc_hard = roc_auc_score(y_test, y_pred)
        check("AUC computed from probabilities (not hard labels)",
              len(np.unique(y_proba)) > 2 and np.isclose(recomputed["AUC"], comparison.loc[name, "AUC"], atol=5e-5),
              f"proba AUC={recomputed['AUC']:.4f} vs hard-label AUC={auc_hard:.4f}")

        # Reproducibility: refit from scratch and compare.
        refit = build_model_pipeline(fresh_classifiers[key]).fit(X_train, y_train)
        check("Retraining from scratch reproduces identical predictions",
              np.array_equal(refit.predict(X_test), y_pred)
              and np.allclose(refit.predict_proba(X_test)[:, 1], y_proba, rtol=0, atol=1e-12))

        report = REPORTS_DIR / f"{key}.txt"
        cm = pd.read_csv(CONFUSION_DIR / f"{key}.csv", index_col=0).to_numpy()
        tn, fp, fn, tp = ((y_test == 0) & (y_pred == 0)).sum(), ((y_test == 0) & (y_pred == 1)).sum(), \
            ((y_test == 1) & (y_pred == 0)).sum(), ((y_test == 1) & (y_pred == 1)).sum()
        check("Classification report and confusion matrix saved and correct",
              report.exists() and (CONFUSION_DIR / f"{key}.png").exists()
              and np.array_equal(cm, [[tn, fp], [fn, tp]]),
              f"TN={tn} FP={fp} FN={fn} TP={tp}")

    # No hardcoded results: no 3+ decimal literals in the training/evaluation source.
    sources = ["train_models.py", "evaluate_models.py"]
    literals = {s: re.findall(r"(?<![\w.])0\.\d{3,}", (PROJECT_ROOT / "src" / s).read_text())
                for s in sources}
    check("No metric-like numeric literals hardcoded in training/evaluation code",
          not any(literals.values()), str(literals))

    check("Original CSV still unchanged after validation", sha1(RAW_DATA_PATH) == ORIGINAL_SHA1)

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
