"""Validation of the Streamlit app and its logic layer.

Usage (needs streamlit installed):  python -m src.validate_app
- Upload handling is tested by feeding in-memory CSV bytes to the same function
  the app calls (Streamlit's AppTest cannot drive st.file_uploader).
- The full UI is run headlessly with AppTest via the bundled-test-set toggle, with
  every fit()/fit_transform() patched to raise, proving nothing is retrained.
Exits with a non-zero status if any check fails.
"""
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.app_logic import DataValidationError, evaluate_csv, load_models
from src.evaluate_models import METRIC_COLUMNS
from src.preprocessing import PROJECT_ROOT, RAW_DATA_PATH, TEST_DATA_PATH
from src.train_models import COMPARISON_PATH, MODEL_DIR, MODEL_NAMES, PREDICTIONS_PATH

ORIGINAL_SHA1 = "252c7d50740da7988d71174d15184247463d975c"
APP_PATH = str(PROJECT_ROOT / "app.py")
results = []


def check(name, passed, detail=""):
    results.append(bool(passed))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def sha1(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()


def to_bytes(df, **kwargs):
    return df.to_csv(index=False, **kwargs).encode()


def expect_error(name, data, filename, models, needle):
    try:
        evaluate_csv(data, filename, models)
        check(name, False, "no error raised")
    except DataValidationError as exc:
        text = " ".join([exc.title, *exc.problems])
        check(name, needle.lower() in text.lower(), text[:110])


def expect_reference(name, data, models, reference, note_needle):
    """Upload must evaluate to exactly the reference metrics and carry a note containing note_needle."""
    try:
        r = evaluate_csv(data, "edge.csv", models)
    except DataValidationError as exc:
        check(name, False, f"rejected: {exc.title}")
        return
    diff = (r["comparison"] - reference).abs().max().max()
    notes = " ".join(m for _, m in r["notes"])
    check(name, diff < 1e-12 and note_needle.lower() in notes.lower()
          and "not seen in training" not in notes, f"max abs diff={diff}")


def notes_of(r, level=None):
    return " ".join(m for lvl, m in r["notes"] if level in (None, lvl))


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


def fresh_app():
    """New AppTest run with Streamlit caches cleared, so patched model loaders take effect."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_resource.clear()
    st.cache_data.clear()
    return AppTest.from_file(APP_PATH, default_timeout=180).run()


def tab_text(tab):
    return " ".join(m.value for m in tab.markdown) + " " + " ".join(c.value for c in tab.caption)


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
    live = result["comparison"]
    subset = evaluate_csv(to_bytes(base.sample(250, random_state=0)), "subset.csv", models)
    check("Different row count (250 rows) evaluates", subset["n_rows"] == 250
          and subset["comparison"].notna().all().all())
    shuffled = base[base.columns[::-1]].assign(customerID="X", extra_col=1)
    r = evaluate_csv(to_bytes(shuffled), "shuffled.csv", models)
    check("Reordered columns + customerID + extra column accepted",
          (r["comparison"] - live).abs().max().max() < 1e-12
          and "customerID" in r["predictions"].columns
          and any("extra" in m for _, m in r["notes"]))
    blanks = pd.concat([raw[raw["TotalCharges"].str.strip() == ""].drop(columns="customerID"), base.head(50)])
    r = evaluate_csv(to_bytes(blanks), "blanks.csv", models)
    zeros = evaluate_csv(to_bytes(blanks.assign(TotalCharges=blanks["TotalCharges"].replace(" ", "0"))),
                         "zeros.csv", models)
    check("All 11 blank TotalCharges rows evaluated exactly as TotalCharges = 0 (preprocessing rule unchanged)",
          r["n_rows"] == 61 and all(np.array_equal(r["details"][k]["y_proba"], zeros["details"][k]["y_proba"])
                                    for k in MODEL_NAMES)
          and not any("TotalCharges" in m for _, m in r["notes"]), f"{r['n_rows']} rows")
    late = blanks.copy()
    late.iloc[0, late.columns.get_loc("tenure")] = 5
    r = evaluate_csv(to_bytes(late), "late.csv", models)
    check("Blank TotalCharges with tenure > 0 still produces a warning",
          "1 blank TotalCharges row(s) have tenure > 0" in notes_of(r, "warning"))
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
    expect_error("Binary/garbage content rejected", bytes(range(256)) * 4, "g.csv", models, "plain-text")

    # Edge cases added with the input-validation layer.
    for col, value in [("tenure", "inf"), ("TotalCharges", "-inf"), ("MonthlyCharges", "1e308")]:
        bad = base.copy().astype({col: object})
        bad.loc[[1, 7], col] = value
        expect_error(f"Non-finite/huge {col} ({value}) rejected", to_bytes(bad), "n.csv", models,
                     f"{col} has infinite or implausibly large values")
    expect_error("Duplicate column after trimming spaces rejected",
                 to_bytes(base.assign(**{" tenure ": base["tenure"]})), "d.csv", models,
                 "'tenure' appears 2 times")
    expect_error("Exact duplicate column rejected",
                 to_bytes(pd.concat([base, base[["Churn"]]], axis=1)), "d.csv", models,
                 "'Churn' appears 2 times")
    expect_error("Semicolon-separated file rejected with delimiter message",
                 to_bytes(base, sep=";"), "s.csv", models, "semicolons")
    expect_error("Tab-separated file rejected with delimiter message",
                 to_bytes(base, sep="\t"), "t.csv", models, "tabs")
    lines = test_bytes.split(b"\n")
    lines[5] += b",extra,values"
    expect_error("Row with too many values rejected with line number", b"\n".join(lines), "r.csv", models,
                 "Expected 20 fields in line 6")
    expect_error("Every row longer than header rejected (no silent column shift)",
                 b"\n".join(l + b",x" if i and l else l for i, l in enumerate(test_bytes.split(b"\n"))),
                 "r.csv", models, "more values than the header")
    expect_error("UTF-16 file rejected with re-save advice", test_bytes.decode().encode("utf-16"),
                 "u.csv", models, "CSV UTF-8")
    expect_error("Wrong-case column name rejected with a hint",
                 to_bytes(base.rename(columns={"tenure": "Tenure"})), "c.csv", models, "Found 'Tenure'")
    expect_reference("SeniorCitizen 0.0/1.0 normalised to training categories",
                     to_bytes(base.astype({"SeniorCitizen": float})), models, live, "SeniorCitizen values")
    expect_reference("SeniorCitizen Yes/No normalised to training categories",
                     to_bytes(base.assign(SeniorCitizen=base["SeniorCitizen"].map({0: "No", 1: "Yes"}))),
                     models, live, "SeniorCitizen values")
    mixed = base["Churn"].where(base.index % 2 == 0, base["Churn"].str.upper()).where(
        base.index % 3 != 0, base["Churn"].str.lower())
    expect_reference("Churn labels in any capitalisation accepted", to_bytes(base.assign(Churn=mixed)),
                     models, live, "capitalisation")
    expect_reference("UTF-8 file with BOM accepted", b"\xef\xbb\xbf" + test_bytes, models, live, "")
    latin = base.assign(customerID=[f"Jos\xe9-{i}" for i in range(len(base))])
    expect_reference("Latin-1 file accepted after UTF-8 fails", latin.to_csv(index=False).encode("latin-1"),
                     models, live, "Latin-1")
    neg = base.copy()
    neg.loc[[0, 1], "tenure"] = -5
    r = evaluate_csv(to_bytes(neg), "neg.csv", models)
    check("Negative tenure evaluated unchanged with a warning",
          "tenure is negative in 2 row(s)" in notes_of(r, "warning") and r["n_rows"] == len(base))
    blank_cat = base.copy()
    blank_cat.loc[[0, 1], "Contract"] = None
    r = evaluate_csv(to_bytes(blank_cat), "blank.csv", models)
    check("Blank category reported as (blank)", "Contract: 2 row(s)" in notes_of(r, "warning")
          and "(blank)" in notes_of(r, "warning"))
    odd = base.copy().astype({"SeniorCitizen": object})
    odd.loc[0, "SeniorCitizen"] = "maybe"
    r = evaluate_csv(to_bytes(odd), "odd.csv", models)
    check("Unrecognised SeniorCitizen value still flagged as unseen (not silently mapped)",
          "SeniorCitizen: 1 row(s) with categories not seen in training ('maybe')" in notes_of(r, "warning"))

    # Missing and broken model files.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for path in MODEL_DIR.glob("*.joblib"):
            shutil.copy(path, tmp / path.name)
        (tmp / "knn.joblib").unlink()
        (tmp / "svm.joblib").write_bytes(b"not a model")
        partial, partial_errors = load_models(tmp)
    check("Missing and broken model files reported; the others still load",
          sorted(partial) == ["decision_tree", "logistic_regression", "naive_bayes"]
          and any("knn.joblib" in e and "not found" in e for e in partial_errors)
          and any("Support Vector Machine" in e and "could not be loaded" in e for e in partial_errors),
          "; ".join(partial_errors)[:110])

    # Full UI, headless, with every fit()/fit_transform() patched to raise.
    patches = [mock.patch.object(cls, attr, fit_guard)
               for cls in all_estimator_classes(models)
               for attr in ("fit", "fit_transform") if hasattr(cls, attr)]
    for p in patches:
        p.start()
    try:
        at = fresh_app()
        check("App starts (welcome screen, no exceptions)",
              not at.exception and any("Get started" in i.value for i in at.info))
        check("No permanent 'models loaded' success message", len(at.success) == 0)
        page_text = " ".join(m.value for m in at.markdown)
        check("Title banner: heading plus roll number only (no subtitle or badges)",
              "Telco Customer Churn Prediction</h1><span class=\"roll\">Roll No. 2026AJ05797</span></div>" in page_text
              and "Classification Dashboard" not in page_text and "pill" not in page_text)
        check("Welcome: no 'Available models' cards; expected CSV format kept",
              [s.value for s in at.subheader] == ["Expected CSV format"] and "model-card" not in page_text)
        check("Evaluation-data note shown in the sidebar under the upload controls",
              any("Use unseen labelled test data" in c.value for c in at.sidebar.caption))
        check("Sidebar starts with '1 · Test data' (no Controls heading or subtitle)",
              at.sidebar.markdown[0].value == "#### 1 · Test data"
              and not any("Controls" in m.value for m in at.sidebar.markdown)
              and not any("ML Assignment 2" in c.value for c in at.sidebar.caption))
        at.toggle[0].set_value(True).run()
        check("App evaluates bundled test CSV without exceptions or errors",
              not at.exception and not at.error, f"exceptions={len(at.exception)}")
        metrics = {m.label: m.value for m in at.metric}
        check("Overview shows record count and churn distribution",
              metrics.get("Records") == "1,409" and metrics.get("Churn = Yes") == "374"
              and metrics.get("Churn = No") == "1,035")
        overview_cards = [m for m in at.metric if m.label in ("Records", "Churn = Yes", "Churn = No")]
        check("Overview cards have no percentage badges; no blank-TotalCharges notice",
              len(overview_cards) == 3 and not any(m.delta for m in overview_cards)
              and not any("TotalCharges" in i.value for i in at.info))
        labels = [t.label for t in at.tabs]
        check("Tabs ordered Inspect / Compare / Predictions (Inspect is the default first tab)",
              [l.split()[-1] for l in labels] == ["model", "models", "Predictions"]
              and "Inspect selected model" in labels[0], " | ".join(labels))
        detail_tab, compare_tab = at.tabs[0], at.tabs[1]
        check("Default tab shows the model chosen in the sidebar",
              any(m.value == "### Logistic Regression" for m in detail_tab.markdown))
        shown = {m: float(metrics[label]) for m, label in
                 [("Accuracy", "Accuracy"), ("AUC", "AUC (ROC)"), ("Precision", "Precision"),
                  ("Recall", "Recall"), ("F1", "F1 score"), ("MCC", "MCC")]}
        check("Six metric cards for default model match Phase 3",
              all(abs(shown[m] - reference.loc["Logistic Regression", m]) < 1e-9 for m in shown), str(shown))
        check("Selected-model tab: exactly one classification report, no separate TN/FP/FN/TP list",
              len(detail_tab.dataframe) == 1 and not any("Plain-text" in e.label for e in at.expander)
              and not any(f"**{c}**" in tab_text(detail_tab) for c in ("TN", "FP", "FN", "TP"))
              and "Rows are the actual class, columns the predicted class." in tab_text(detail_tab))
        check("No correct/incorrect duplicate cards or outcomes table",
              "Correct predictions" not in metrics and "Incorrect predictions" not in metrics)
        comparison_table = compare_tab.dataframe[0].value
        check("Comparison table: all five models, exactly the six metrics (no rank column)",
              list(comparison_table.index) == list(MODEL_NAMES.values())
              and list(comparison_table.columns) == METRIC_COLUMNS)
        glossary = tab_text(compare_tab)
        check("Comparison tab: no 'What the metrics mean' section; bold/tie-break note kept",
              "What the metrics mean" not in glossary and "Proportion of all predictions" not in glossary
              and "Bold = best value in each column. MCC ties are broken by AUC." in glossary)
        check("Best-model summary kept; metric chart and its controls removed",
              any("Best model on this data" in m.value for m in compare_tab.markdown)
              and len(at.radio) == 0 and not compare_tab.get("arrow_vega_lite_chart")
              and "by model" not in glossary and "Dark bar" not in glossary)

        before_report = detail_tab.dataframe[0].value.copy()
        at.selectbox[0].set_value("Gaussian Naive Bayes").run()
        detail_tab = at.tabs[0]
        metrics = {m.label: m.value for m in at.metric}
        after_report = detail_tab.dataframe[0].value
        check("Sidebar model change updates the default tab heading and metric cards",
              any(m.value == "### Gaussian Naive Bayes" for m in detail_tab.markdown)
              and abs(float(metrics["Recall"]) - reference.loc["Gaussian Naive Bayes", "Recall"]) < 1e-9)
        check("Classification report updates on model change",
              not before_report.equals(after_report)
              and abs(after_report.loc["Yes", "Recall"] - reference.loc["Gaussian Naive Bayes", "Recall"]) < 5e-5)
        check("Predictions tab offers the download", len(at.tabs[2].get("download_button")) == 1)
        check("No fit()/fit_transform() called while running the app", not at.exception,
              f"{len(patches)} methods guarded")
    finally:
        for p in patches:
            p.stop()

    # Model-loading failures surface clearly in the UI.
    partial_models = {k: models[k] for k in ("logistic_regression", "decision_tree", "naive_bayes")}
    load_errors = ["K-Nearest Neighbors: file not found (knn.joblib)",
                   "Support Vector Machine: could not be loaded (UnpicklingError: invalid load key)"]
    with mock.patch("src.app_logic.load_models", return_value=(partial_models, load_errors)):
        at = fresh_app()
        at.toggle[0].set_value(True).run()
        errors_shown = " ".join(e.value for e in at.error)
        check("Partly failed model loading: clear errors, remaining models still evaluated",
              not at.exception and "knn.joblib" in errors_shown and "Support Vector Machine" in errors_shown
              and len(at.tabs[1].dataframe[0].value) == 3)
    with mock.patch("src.app_logic.load_models", return_value=({}, load_errors)):
        at = fresh_app()
        check("No models loaded: clear error and evaluation stops",
              not at.exception and any("No trained models could be loaded" in e.value for e in at.error)
              and len(at.tabs) == 0)
    fresh_app()  # leave the caches holding the real models

    check("Model files unchanged", model_hashes == {p.name: sha1(p) for p in sorted(MODEL_DIR.glob("*.joblib"))})
    check("Original CSV unchanged", sha1(RAW_DATA_PATH) == ORIGINAL_SHA1)
    app_source = (PROJECT_ROOT / "app.py").read_text() + (PROJECT_ROOT / "src" / "app_logic.py").read_text()
    check("No fit() calls in app source", ".fit(" not in app_source and "fit_transform" not in app_source)

    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
