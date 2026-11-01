"""Train and evaluate the five baseline classifiers (Phase 3).

Usage: python3 -m src.train_models

Each model is Pipeline(preprocessor -> classifier), fitted on the Phase 2 training
split only and evaluated on test_data.csv (project root). Saved pipelines accept
raw feature columns, so the future Streamlit app can load test CSVs directly.
"""
import platform
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluate_models import (
    METRIC_COLUMNS, classification_report_text, confusion_matrix_frame,
    evaluate_model, plot_confusion_matrix,
)
from src.preprocessing import (
    PROJECT_ROOT, RANDOM_STATE, TEST_DATA_PATH, build_preprocessor, clean_features,
    load_train_test_split, split_features_target,
)

MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "src" / "results"
COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"
PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.csv"
REPORTS_DIR = RESULTS_DIR / "classification_reports"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
SUMMARY_PATH = RESULTS_DIR / "metrics_summary.txt"

# File-name key -> display name, in reporting order.
MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "K-Nearest Neighbors",
    "naive_bayes": "Gaussian Naive Bayes",
    "svm": "Support Vector Machine",
}
# Overall ranking metric: MCC uses all four confusion-matrix cells and is robust
# to the 73/27 class imbalance; AUC breaks ties.
RANKING_METRICS = ["MCC", "AUC"]


def build_classifiers():
    """Fresh, unfitted baseline classifiers (assignment-specified settings)."""
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "naive_bayes": GaussianNB(),
        "svm": SVC(probability=True, random_state=RANDOM_STATE),
    }


def build_model_pipeline(classifier):
    return Pipeline(steps=[("preprocessor", build_preprocessor()),
                           ("classifier", classifier)])


def load_datasets():
    """Return X_train, y_train (Phase 2 split) and X_test, y_test (saved test CSV).

    The split is regenerated with the fixed Phase 2 settings; this stops if the
    regenerated test rows differ from test_data.csv.
    """
    X_train, X_test_split, y_train, y_test_split = load_train_test_split()
    X_test, y_test = split_features_target(pd.read_csv(TEST_DATA_PATH))
    same_features = clean_features(X_test_split).reset_index(drop=True).equals(clean_features(X_test))
    same_target = y_test_split.reset_index(drop=True).equals(y_test)
    if not (same_features and same_target):
        raise RuntimeError(f"{TEST_DATA_PATH} does not match the Phase 2 split; "
                           "rerun python3 -m src.preprocessing")
    return X_train, y_train, X_test, y_test


def train_models(X_train, y_train):
    """Fit one pipeline per classifier on the training split; capture warnings."""
    fitted, fit_warnings = {}, {}
    for key, classifier in build_classifiers().items():
        pipeline = build_model_pipeline(classifier)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(X_train, y_train)
        fitted[key] = pipeline
        fit_warnings[key] = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
    return fitted, fit_warnings


def evaluate_and_save(fitted, X_test, y_test):
    """Evaluate every pipeline on the test set and write all result files."""
    for directory in (MODEL_DIR, REPORTS_DIR, CONFUSION_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    rows, reports, matrices = [], {}, {}
    predictions = pd.DataFrame({"y_true": y_test})
    for key, pipeline in fitted.items():
        name = MODEL_NAMES[key]
        metrics, y_pred, y_proba = evaluate_model(pipeline, X_test, y_test)
        rows.append({"Model": name, **metrics})
        predictions[f"{key}_pred"] = y_pred
        predictions[f"{key}_proba"] = y_proba

        reports[key] = classification_report_text(y_test, y_pred)
        (REPORTS_DIR / f"{key}.txt").write_text(
            f"{name} - classification report (test set, n={len(y_test)})\n"
            f"Positive class: Churn = Yes (1)\n\n{reports[key]}")

        matrices[key] = confusion_matrix_frame(y_test, y_pred)
        matrices[key].to_csv(CONFUSION_DIR / f"{key}.csv")
        fig = plot_confusion_matrix(matrices[key], f"{name}\nConfusion matrix (test set, n={len(y_test):,})",
                                    CONFUSION_DIR / f"{key}.png")
        plt.close(fig)

        joblib.dump(pipeline, MODEL_DIR / f"{key}.joblib")

    comparison = pd.DataFrame(rows).set_index("Model")[METRIC_COLUMNS]
    comparison.round(4).to_csv(COMPARISON_PATH)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    return comparison, reports, matrices


def write_summary(comparison, reports, matrices, fit_warnings, fitted, y_train, y_test):
    ranked = comparison.sort_values(RANKING_METRICS, ascending=False)
    best = ranked.index[0]
    lines = [
        "TELCO CUSTOMER CHURN - PHASE 3 BASELINE MODEL RESULTS",
        "=" * 72,
        f"Train rows: {len(y_train)} (churn rate {y_train.mean():.4f})   "
        f"Test rows: {len(y_test)} (churn rate {y_test.mean():.4f})",
        "Split: 80/20 stratified, random_state=42 (Phase 2); test set = test_data.csv",
        "Positive class: Churn = Yes (1). Precision/Recall/F1 use binary averaging, zero_division=0.",
        "AUC uses predict_proba() probabilities for the positive class.",
        f"Environment: Python {platform.python_version()}, scikit-learn {sklearn.__version__}",
        "",
        "MODEL CONFIGURATIONS (each wrapped as Pipeline(preprocessor -> classifier))",
        "-" * 72,
    ]
    for key, pipeline in fitted.items():
        lines.append(f"{MODEL_NAMES[key]:<24} {pipeline.named_steps['classifier']!r}")

    lines += ["", "METRICS ON HELD-OUT TEST SET (rounded to 4 d.p.)", "-" * 72,
              comparison.round(4).to_string(), "",
              f"RANKING (by {RANKING_METRICS[0]}, ties broken by {RANKING_METRICS[1]})", "-" * 72]
    for position, (name, row) in enumerate(ranked.iterrows(), 1):
        lines.append(f"{position}. {name:<24} MCC={row['MCC']:.4f}  AUC={row['AUC']:.4f}  F1={row['F1']:.4f}")
    lines += ["", "BEST MODEL PER METRIC", "-" * 72]
    for metric in METRIC_COLUMNS:
        lines.append(f"{metric:<10} {comparison[metric].idxmax():<24} {comparison[metric].max():.4f}")
    lines += ["", f"Overall best baseline (by {RANKING_METRICS[0]}): {best}", ""]

    lines += ["CONFUSION MATRICES (rows = actual, columns = predicted)", "-" * 72]
    for key, matrix in matrices.items():
        lines += [MODEL_NAMES[key], matrix.to_string(), ""]

    lines += ["CLASSIFICATION REPORTS", "-" * 72]
    for key, report in reports.items():
        lines += [MODEL_NAMES[key], report]

    lines += ["WARNINGS RAISED DURING FITTING", "-" * 72]
    for key, messages in fit_warnings.items():
        lines.append(f"{MODEL_NAMES[key]}: {'; '.join(messages) if messages else 'none'}")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")


def main():
    X_train, y_train, X_test, y_test = load_datasets()
    print(f"Training on {len(X_train)} rows, evaluating on {len(X_test)} rows")
    fitted, fit_warnings = train_models(X_train, y_train)
    comparison, reports, matrices = evaluate_and_save(fitted, X_test, y_test)
    write_summary(comparison, reports, matrices, fit_warnings, fitted, y_train, y_test)
    print(comparison.round(4).to_string())
    for key, messages in fit_warnings.items():
        for message in messages:
            print(f"WARNING [{MODEL_NAMES[key]}]: {message}")
    print(f"\nSaved models to {MODEL_DIR} and results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
