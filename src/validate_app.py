"""Phase 4 validation of the Streamlit app and its logic layer.

Usage (needs streamlit installed):  python -m src.validate_app
- Upload handling is tested by feeding in-memory CSV bytes to the same function
  the app calls (Streamlit's AppTest cannot drive st.file_uploader).
- The full UI is run headlessly with AppTest via the bundled-test-set toggle, with
  every fit()/fit_transform() patched to raise, proving nothing is retrained.
Exits with a non-zero status if any check fails.
"""
import hashlib
import sys
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.app_logic import DataValidationError, evaluate_csv, load_models
from src.preprocessing import PROJECT_ROOT, RAW_DATA_PATH, TEST_DATA_PATH
from src.train_models import COMPARISON_PATH, MODEL_DIR, MODEL_NAMES, PREDICTIONS_PATH

ORIGINAL_SHA1 = "252c7d50740da7988d71174d15184247463d975c"
results = []


def check(name, passed, detail=""):
    results.append(bool(passed))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def sha1(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()


def to_bytes(df):
    return df.to_csv(index=False).encode()


def expect_error(name, data, filename, models, needle):
    try:
        evaluate_csv(data, filename, models)
        check(name, False, "no error raised")
    except DataValidationError as exc:
        text = " ".join([exc.title, *exc.problems])
        check(name, needle.lower() in text.lower(), text[:110])


def fit_guard(*args, **kwargs):
    raise AssertionError("fit()/fit_transform() was called by the app")


def all_estimator_classes(models):
    """Every estimator class inside the saved pipelines (pipeline, transformers, classifiers)."""
    classes = set()

    def walk(obj):
        if isinstance(obj, BaseEstimator):
            classes.add(type(obj))
            for value in vars(obj).values():
                for item in value if isinstance(value, (list, tuple)) else [value]:
                    for leaf in item if isinstance(item, tuple) else [item]:
                        walk(leaf)

    for model in models.values():
        walk(model)
    return classes


def main():
    model_hashes = {p.name: sha1(p) for p in sorted(MODEL_DIR.glob("*.joblib"))}
    models, errors = load_models()
    check("All five saved models load", len(models) == 5 and not errors, ", ".join(models))

    # Original test CSV reproduces Phase 3 exactly.
    test_bytes = TEST_DATA_PATH.read_bytes()
    result = evaluate_csv(test_bytes, "test_data.csv", models)
    reference = pd.read_csv(COMPARISON_PATH, index_col="Model")
    diff = (result["comparison"].round(4) - reference).abs().max().max()
    check("Original test CSV: 5 x 6 metrics match Phase 3 model_comparison.csv", diff == 0,
          f"max abs diff={diff}")
    stored = pd.read_csv(PREDICTIONS_PATH)
    same_preds = all(np.array_equal(result["details"][k]["y_pred"], stored[f"{k}_pred"])
                     and np.allclose(result["details"][k]["y_proba"], stored[f"{k}_proba"], atol=1e-12)
                     for k in MODEL_NAMES)
    check("Original test CSV: every model's predictions match Phase 3", same_preds)
    check("Best model by MCC identified", result["best"] == reference["MCC"].idxmax(), result["best"])
    expected_cols = ["row", "actual_churn"] + [f"{k}_{s}" for k in MODEL_NAMES
                                               for s in ("predicted", "churn_probability")]
    preds = result["predictions"]
    check("Predictions download has actual + predicted label/probability per model",
          list(preds.columns) == expected_cols and len(preds) == result["n_rows"],
          f"{preds.shape[1]} columns, {len(preds)} rows")
    check("Predicted labels in download are Yes/No and match model output",
          set(preds["actual_churn"]) <= {"Yes", "No"}
          and (preds["svm_predicted"].map({"No": 0, "Yes": 1}).to_numpy()
               == result["details"]["svm"]["y_pred"]).all())

    # Upload variants.
    base = pd.read_csv(TEST_DATA_PATH)
    raw = pd.read_csv(RAW_DATA_PATH)
    subset = evaluate_csv(to_bytes(base.sample(250, random_state=0)), "subset.csv", models)
    check("Different row count (250 rows) evaluates", subset["n_rows"] == 250
          and subset["comparison"].notna().all().all())
    shuffled = base[base.columns[::-1]].assign(customerID="X", extra_col=1)
    r = evaluate_csv(to_bytes(shuffled), "shuffled.csv", models)
    check("Reordered columns + customerID + extra column accepted",
          (r["comparison"] - result["comparison"]).abs().max().max() < 1e-12
          and "customerID" in r["predictions"].columns
          and any("extra" in m for _, m in r["notes"]))
    blanks = raw[raw["TotalCharges"].str.strip() == ""].drop(columns="customerID")
    r = evaluate_csv(to_bytes(pd.concat([blanks, base.head(50)])), "blanks.csv", models)
    check("All 11 blank TotalCharges rows handled by preprocessing (set to 0)",
          any("11 blank TotalCharges" in m for _, m in r["notes"]), f"{r['n_rows']} rows")
    unseen = base.head(40).copy()
    unseen.loc[:4, "PaymentMethod"] = "Cryptocurrency"
    r = evaluate_csv(to_bytes(unseen), "unseen.csv", models)
    check("Unseen categorical values evaluate with a warning",
          any("Cryptocurrency" in m for _, m in r["notes"]) and r["comparison"].notna().all().all())
    only_no = base[base["Churn"] == "No"].head(30)
    r = evaluate_csv(to_bytes(only_no), "only_no.csv", models)
    check("Single-class upload: AUC shown as undefined, other metrics computed",
          r["comparison"]["AUC"].isna().all()
          and r["comparison"].drop(columns="AUC").notna().all().all()
          and (r["comparison"]["Precision"] >= 0).all())

    expect_error("Missing column rejected", to_bytes(base.drop(columns=["Contract", "tenure"])),
                 "m.csv", models, "Missing: tenure, Contract")
    expect_error("Missing Churn column rejected", to_bytes(base.drop(columns=["Churn"])),
                 "m.csv", models, "Churn")
    bad = base.head(20).copy()
    bad.loc[0, "Churn"] = "Maybe"
    expect_error("Unexpected Churn label rejected", to_bytes(bad), "b.csv", models, "'Maybe'")
    bad = base.head(20).copy()
    bad.loc[3, "Churn"] = None
    expect_error("Blank Churn label rejected", to_bytes(bad), "b.csv", models, "Churn is blank")
    bad = base.head(20).copy()
    bad["MonthlyCharges"] = bad["MonthlyCharges"].astype(object)
    bad.loc[2, "MonthlyCharges"] = "abc"
    expect_error("Non-numeric MonthlyCharges rejected", to_bytes(bad), "b.csv", models, "MonthlyCharges")
    expect_error("Empty file rejected", b"", "e.csv", models, "empty")
    expect_error("Header-only file rejected", to_bytes(base.head(0)), "h.csv", models, "no data rows")
    expect_error("Non-CSV filename rejected", test_bytes, "data.xlsx", models, ".csv")
    expect_error("Binary/garbage content rejected", bytes(range(256)) * 4, "g.csv", models, "")

    # Full UI, headless, with every fit()/fit_transform() patched to raise.
    from streamlit.testing.v1 import AppTest

    patches = [mock.patch.object(cls, attr, fit_guard)
               for cls in all_estimator_classes(models)
               for attr in ("fit", "fit_transform") if hasattr(cls, attr)]
    for p in patches:
        p.start()
    try:
        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=180).run()
        check("App starts (welcome screen, no exceptions)",
              not at.exception and any("Get started" in i.value for i in at.info))
        at.toggle[0].set_value(True).run()
        check("App evaluates bundled test CSV without exceptions or errors",
              not at.exception and not at.error, f"exceptions={len(at.exception)}")
        metrics = {m.label: m.value for m in at.metric}
        check("Overview shows record count and churn distribution",
              metrics.get("Records") == "1,409" and metrics.get("Churn = Yes") == "374")
        comparison_table = at.dataframe[0].value
        check("Comparison table lists all five models with all six metrics",
              list(comparison_table.index) == list(MODEL_NAMES.values())
              and all(m in comparison_table.columns for m in reference.columns))
        shown = {m: float(metrics[label]) for m, label in
                 [("Accuracy", "Accuracy"), ("AUC", "AUC (ROC)"), ("Precision", "Precision"),
                  ("Recall", "Recall"), ("F1", "F1 score"), ("MCC", "MCC")]}
        check("Six metric cards for default model match Phase 3",
              all(abs(shown[m] - reference.loc["Logistic Regression", m]) < 1e-9 for m in shown), str(shown))

        before_report = at.dataframe[2].value.copy()
        before_cm = (metrics["Correct predictions"], metrics["Incorrect predictions"])
        at.selectbox[0].set_value("Gaussian Naive Bayes").run()
        metrics = {m.label: m.value for m in at.metric}
        after_cm = (metrics["Correct predictions"], metrics["Incorrect predictions"])
        after_report = at.dataframe[2].value
        check("Confusion-matrix counts update on model change", before_cm != after_cm,
              f"LR {before_cm} -> NB {after_cm}")
        check("Classification report updates on model change",
              not before_report.equals(after_report)
              and abs(after_report.loc["Yes", "Recall"] - reference.loc["Gaussian Naive Bayes", "Recall"]) < 5e-5)
        check("No fit()/fit_transform() called while running the app", not at.exception,
              f"{len(patches)} methods guarded")
    finally:
        for p in patches:
            p.stop()

    check("Model files unchanged", model_hashes == {p.name: sha1(p) for p in sorted(MODEL_DIR.glob("*.joblib"))})
    check("Original CSV unchanged", sha1(RAW_DATA_PATH) == ORIGINAL_SHA1)
    app_source = (PROJECT_ROOT / "app.py").read_text() + (PROJECT_ROOT / "src" / "app_logic.py").read_text()
    check("No fit() calls in app source", ".fit(" not in app_source and "fit_transform" not in app_source)

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
