"""Reusable evaluation helpers: the six metrics, classification reports and confusion matrices.

Positive class is Churn = Yes (encoded as 1). Shared by training, validation and
the future Streamlit app.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

POSITIVE_LABEL = 1
CLASS_LABELS = [0, 1]
CLASS_NAMES = ["No", "Yes"]
METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def positive_class_proba(model, X):
    """Probability of Churn = Yes, located via classes_ rather than assuming column 1."""
    column = list(model.classes_).index(POSITIVE_LABEL)
    return model.predict_proba(X)[:, column]


def compute_metrics(y_true, y_pred, y_proba):
    """Return the six assignment metrics; AUC uses positive-class probabilities.

    AUC is undefined when y_true contains a single class; it is returned as NaN
    instead of raising, so uploads with only churners or non-churners still evaluate.
    """
    binary = dict(pos_label=POSITIVE_LABEL, average="binary", zero_division=0)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba) if both_classes else float("nan"),
        "Precision": precision_score(y_true, y_pred, **binary),
        "Recall": recall_score(y_true, y_pred, **binary),
        "F1": f1_score(y_true, y_pred, **binary),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def evaluate_model(model, X, y):
    """Predict on X and return (metrics, y_pred, y_proba)."""
    y_pred = model.predict(X)
    y_proba = positive_class_proba(model, X)
    return compute_metrics(y, y_pred, y_proba), y_pred, y_proba


def classification_report_text(y_true, y_pred):
    return classification_report(y_true, y_pred, labels=CLASS_LABELS,
                                 target_names=CLASS_NAMES, digits=4, zero_division=0)


def classification_report_frame(y_true, y_pred):
    """Classification report as a DataFrame: rows No, Yes, macro avg, weighted avg."""
    report = classification_report(y_true, y_pred, labels=CLASS_LABELS, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    accuracy = report.pop("accuracy", None)
    frame = pd.DataFrame(report).T.rename(columns={"precision": "Precision", "recall": "Recall",
                                                   "f1-score": "F1", "support": "Support"})
    frame["Support"] = frame["Support"].astype(int)
    frame.attrs["accuracy"] = accuracy
    return frame


def confusion_matrix_frame(y_true, y_pred):
    """2x2 confusion matrix with rows = actual class, columns = predicted class."""
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    return pd.DataFrame(matrix,
                        index=[f"Actual {name}" for name in CLASS_NAMES],
                        columns=[f"Predicted {name}" for name in CLASS_NAMES])


def plot_confusion_matrix(cm_frame, title, path=None, cell_names=False):
    """Draw counts plus the share of each actual class; save to path if given.

    cell_names=True also labels the cells TN / FP / FN / TP (used by the app).
    """
    counts = cm_frame.to_numpy()
    row_share = counts / counts.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("blue_ramp", ["#f0f6fe", "#86b6ef", "#184f95"])
    names = [["TN", "FP"], ["FN", "TP"]]

    fig, ax = plt.subplots(figsize=(5.6, 4.8), facecolor="#fcfcfb")
    ax.imshow(row_share, cmap=cmap, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            prefix = f"{names[i][j]}\n" if cell_names else ""
            ax.text(j, i, f"{prefix}{counts[i, j]:,}\n({row_share[i, j]:.1%} of actual)",
                    ha="center", va="center", fontsize=11,
                    color="white" if row_share[i, j] > 0.55 else "#0b0b0b")
    ax.set_xticks([0, 1], ["No (stay)", "Yes (churn)"])
    ax.set_yticks([0, 1], ["No (stayed)", "Yes (churned)"])
    ax.set_xlabel("Predicted label", color="#52514e")
    ax.set_ylabel("Actual label", color="#52514e")
    ax.set_title(title, fontsize=12, fontweight="semibold", pad=10)
    ax.tick_params(colors="#52514e", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    return fig
