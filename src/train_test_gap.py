"""Read-only diagnostic: score the saved pipelines on the training split and the test set.

Usage: python3 -m src.train_test_gap
Nothing is fitted; the saved models are only used for prediction. A large gap
between training and test scores indicates overfitting. A small gap on one split
does not prove a model generalises; it only fails to show overfitting here.
"""
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluate_models import evaluate_model
from src.train_models import MODEL_DIR, MODEL_NAMES, RESULTS_DIR, load_datasets

GAP_PATH = RESULTS_DIR / "train_test_gap.csv"
REPORT_PATH = RESULTS_DIR / "train_test_gap.txt"
GAP_METRICS = ["Accuracy", "F1", "MCC"]


def main():
    X_train, y_train, X_test, y_test = load_datasets()
    rows, notes = [], []
    for key, name in MODEL_NAMES.items():
        model = joblib.load(MODEL_DIR / f"{key}.joblib")
        train_metrics, _, _ = evaluate_model(model, X_train, y_train)
        test_metrics, _, _ = evaluate_model(model, X_test, y_test)
        row = {"Model": name}
        for metric in GAP_METRICS:
            row[f"Train {metric}"] = train_metrics[metric]
            row[f"Test {metric}"] = test_metrics[metric]
            row[f"{metric} gap"] = train_metrics[metric] - test_metrics[metric]
        rows.append(row)

        classifier = model.named_steps["classifier"]
        if key == "decision_tree":
            notes.append(f"Decision Tree: depth={classifier.get_depth()}, leaves={classifier.get_n_leaves()}")
        if key == "svm":
            scores = model.decision_function(X_test)
            notes.append(f"SVM test AUC from decision_function scores: {roc_auc_score(y_test, scores):.4f} "
                         f"(predict_proba AUC: {test_metrics['AUC']:.4f})")

    table = pd.DataFrame(rows).set_index("Model").round(4)
    table.to_csv(GAP_PATH)
    lines = ["TRAIN vs TEST SCORES OF THE SAVED PIPELINES (no refitting)",
             f"Train rows: {len(y_train)}   Test rows: {len(y_test)}", "",
             table.to_string(), "", *notes, ""]
    REPORT_PATH.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
