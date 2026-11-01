"""Streamlit-independent logic for the dashboard: loading saved models, reading and
validating uploaded CSVs, and evaluating every model on the uploaded data.

Nothing here fits a model or transformer: saved pipelines are only used for
predict() / predict_proba() / transform().

Input compatibility fixes (encodings, label capitalisation, SeniorCitizen spellings)
live here, not in src/preprocessing.py: the saved pipelines call
preprocessing.clean_features(), so changing it would change the saved models.
"""
import io
import warnings

import joblib
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from src.evaluate_models import (
    METRIC_COLUMNS, classification_report_frame, compute_metrics, confusion_matrix_frame,
    positive_class_proba,
)
from src.preprocessing import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, ID_COLUMN, NUMERICAL_FEATURES, TARGET, TARGET_MAPPING,
    clean_features,
)
from src.train_models import MODEL_DIR, MODEL_NAMES, RANKING_METRICS

# Short, results-free descriptions shown next to the selected model.
MODEL_DESCRIPTIONS = {
    "logistic_regression": "A linear model that estimates churn probability from a weighted sum of the "
                           "scaled and one-hot encoded features. Fast, stable and easy to interpret.",
    "decision_tree": "Learns a hierarchy of if/else splits on the features. Unlimited depth here, so it can "
                     "fit the training data very closely and may overfit.",
    "knn": "Predicts from the 5 most similar training customers (Euclidean distance on the scaled, "
           "encoded features). No explicit training step; all training rows are stored.",
    "naive_bayes": "Gaussian Naive Bayes assumes features are independent given the class. Very fast; "
                   "tends to flag many customers as churners (high recall, lower precision).",
    "svm": "A support vector machine with an RBF kernel that finds a maximum-margin boundary. "
           "Probabilities come from Platt scaling fitted during training.",
}
MODEL_CONFIGS = {
    "logistic_regression": "LogisticRegression(max_iter=2000, random_state=42)",
    "decision_tree": "DecisionTreeClassifier(random_state=42)",
    "knn": "KNeighborsClassifier(n_neighbors=5)",
    "naive_bayes": "GaussianNB()",
    "svm": "SVC(probability=True, random_state=42)",
}
LABEL_NAMES = {v: k for k, v in TARGET_MAPPING.items()}  # 0 -> "No", 1 -> "Yes"
# Largest magnitude accepted for tenure and charges. Larger values are not plausible and
# overflow inside the models (the decision tree works in float32).
MAX_ABS_NUMERIC = 1e15
# SeniorCitizen is 0/1 in the training data; equivalent spellings map to those categories.
SENIOR_CITIZEN_VALUES = {"0": "0", "1": "1", "0.0": "0", "1.0": "1", "no": "0", "yes": "1",
                         "false": "0", "true": "1", "n": "0", "y": "1"}
OTHER_DELIMITERS = {";": "semicolons (;)", "\t": "tabs", "|": "pipes (|)"}


class DataValidationError(ValueError):
    """Raised for uploads that cannot be evaluated; `problems` lists each issue."""

    def __init__(self, title, problems=()):
        super().__init__(title)
        self.title = title
        self.problems = list(problems)


def load_models(model_dir=MODEL_DIR):
    """Load every saved pipeline. Returns (models by key, list of error messages)."""
    models, errors = {}, []
    for key, name in MODEL_NAMES.items():
        path = model_dir / f"{key}.joblib"
        if not path.exists():
            errors.append(f"{name}: file not found ({path.name})")
            continue
        try:
            models[key] = joblib.load(path)
        except Exception as exc:  # noqa: BLE001 - surface any unpickling problem to the user
            errors.append(f"{name}: could not be loaded ({type(exc).__name__}: {exc})")
    return models, errors


def read_csv_bytes(data, filename="upload.csv"):
    """Parse uploaded bytes into a DataFrame. Returns (df, notes).

    UTF-8 (with or without a BOM) is tried first, then Latin-1. Raises
    DataValidationError for binary files, other delimiters, duplicate column names
    and rows that do not match the header.
    """
    if not str(filename).lower().endswith(".csv"):
        raise DataValidationError("Only .csv files are supported.", [f"Received: {filename}"])
    if not data or not data.strip():
        raise DataValidationError("The uploaded file is empty.")
    if b"\x00" in data:
        raise DataValidationError(
            "The file is not a plain-text CSV.",
            ["It contains binary data, e.g. an Excel workbook or a UTF-16 export. "
             "Save it as 'CSV UTF-8 (comma delimited)' and upload it again."])
    notes = []
    try:
        data.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        encoding = "latin-1"
        notes.append(("info", "The file is not valid UTF-8, so it was read as Latin-1 (ISO-8859-1) text."))

    header = _parse_csv(data, encoding, header=None, nrows=1, dtype=str, keep_default_na=False)
    names = [str(c).strip() for c in header.iloc[0]] if len(header) else []
    if len(names) == 1:
        for sep, label in OTHER_DELIMITERS.items():
            if sep in names[0]:
                raise DataValidationError(
                    f"The file appears to separate columns with {label}, not commas.",
                    ["Save it as a comma-separated CSV (in Excel: 'CSV UTF-8 (comma delimited)') "
                     "and upload it again."])
    counts = pd.Series([n for n in names if n], dtype=object).value_counts()
    duplicated = counts[counts > 1]
    if not duplicated.empty:
        raise DataValidationError(
            "The CSV has duplicate column names.",
            [f"'{name}' appears {count} times" for name, count in duplicated.items()]
            + ["Names are compared after removing leading and trailing spaces; each column must appear once."])

    df = _parse_csv(data, encoding)
    if not isinstance(df.index, pd.RangeIndex):
        # pandas uses the first column as the index when every row has one value more than the header.
        raise DataValidationError(
            "The data rows have more values than the header has column names.",
            ["Check for a trailing comma at the end of each row or a missing column name in the header."])
    df.columns = [str(c).strip() for c in df.columns]
    return df, notes


def _parse_csv(data, encoding, **kwargs):
    try:
        return pd.read_csv(io.BytesIO(data), encoding=encoding, **kwargs)
    except (EmptyDataError, ParserError) as exc:
        detail = str(exc).strip().replace("Error tokenizing data. C error: ", "")
        raise DataValidationError(
            "The file could not be parsed as CSV.",
            [detail, "Every row must have the same number of comma-separated values; "
                     "values that contain commas must be quoted."]) from None


def validate_dataset(df):
    """Check columns, labels and numeric fields. Returns (X, y, notes).

    `notes` is a list of (level, message) tuples with level "info" or "warning".
    Values are never changed except where noted: Churn labels are matched
    case-insensitively and SeniorCitizen spellings are mapped to the 0/1 categories
    the models were trained on.
    """
    if df.empty:
        raise DataValidationError("The CSV has a header but no data rows.")

    missing = [c for c in FEATURE_COLUMNS + [TARGET] if c not in df.columns]
    if missing:
        by_lower = {str(c).lower(): c for c in df.columns}
        hints = [f"Found '{by_lower[c.lower()]}': column names are case-sensitive, so rename it to '{c}'."
                 for c in missing if c.lower() in by_lower]
        raise DataValidationError(
            f"The CSV is missing {len(missing)} required column(s).",
            [f"Missing: {', '.join(missing)}", *hints,
             "Required: the 19 input features plus the Churn target (customerID is optional)."])

    problems, notes = [], []
    label_text = df[TARGET].astype(str).str.strip()
    blank_labels = df[TARGET].isna() | label_text.eq("")
    labels = label_text.str.lower().map({k.lower(): k for k in TARGET_MAPPING})
    unexpected = sorted(set(label_text[~blank_labels & labels.isna()]))
    if blank_labels.any():
        problems.append(f"Churn is blank in {int(blank_labels.sum())} row(s) "
                        f"(e.g. data rows {_row_numbers(blank_labels)}).")
    if unexpected:
        problems.append(f"Unexpected Churn labels: {', '.join(map(repr, unexpected[:10]))}. "
                        "Allowed values are Yes and No (any capitalisation).")

    numbers = {}
    for col in NUMERICAL_FEATURES:
        values = pd.to_numeric(df[col], errors="coerce")
        blank = df[col].isna() | df[col].astype(str).str.strip().eq("")
        if col == "TotalCharges":  # blank TotalCharges is allowed (not yet billed)
            invalid = values.isna() & ~blank
            if invalid.any():
                problems.append(f"TotalCharges is non-numeric in {int(invalid.sum())} row(s) "
                                f"(e.g. data rows {_row_numbers(invalid)}).")
        elif values.isna().any():
            invalid = values.isna()
            problems.append(f"{col} must be numeric but is blank or non-numeric in "
                            f"{int(invalid.sum())} row(s) (e.g. data rows {_row_numbers(invalid)}).")
        too_large = values.notna() & ~(values.abs() <= MAX_ABS_NUMERIC)
        if too_large.any():
            problems.append(f"{col} has infinite or implausibly large values (beyond ±{MAX_ABS_NUMERIC:.0e}) in "
                            f"{int(too_large.sum())} row(s) (e.g. data rows {_row_numbers(too_large)}).")
        numbers[col] = values

    if problems:
        raise DataValidationError("The CSV contains values that cannot be evaluated.", problems)

    if (label_text[~blank_labels] != labels[~blank_labels]).any():
        notes.append(("info", "Churn labels were matched without regard to capitalisation (e.g. 'yes' = 'Yes')."))
    # Blank TotalCharges are set to 0 by clean_features() (not yet billed); only warn when that looks wrong.
    suspicious_blank = numbers["TotalCharges"].isna() & numbers["tenure"].gt(0)
    if suspicious_blank.any():
        notes.append(("warning", f"{int(suspicious_blank.sum())} blank TotalCharges row(s) have tenure > 0, "
                                 "so the standard rule of treating them as 0 may understate their charges."))
    for col, values in numbers.items():
        negative = values.lt(0)
        if negative.any():
            notes.append(("warning", f"{col} is negative in {int(negative.sum())} row(s) (e.g. data rows "
                                     f"{_row_numbers(negative)}). Negative values are not valid for this field; "
                                     "they were evaluated unchanged, so please check the data."))
    extra = [c for c in df.columns if c not in FEATURE_COLUMNS + [TARGET, ID_COLUMN]]
    if extra:
        notes.append(("info", f"Ignored {len(extra)} extra column(s): {', '.join(extra[:8])}"
                              f"{' …' if len(extra) > 8 else ''}"))

    X = df[FEATURE_COLUMNS].copy()
    senior_text = X["SeniorCitizen"].astype(str).str.strip()
    senior = senior_text.str.lower().map(SENIOR_CITIZEN_VALUES)
    converted = senior.notna() & senior_text.ne(senior)
    if converted.any():
        X["SeniorCitizen"] = senior.where(senior.notna(), X["SeniorCitizen"])
        examples = ", ".join(map(repr, sorted(set(senior_text[converted]))[:4]))
        notes.append(("info", f"SeniorCitizen values {examples} were read as the training categories "
                              "0 (No) and 1 (Yes)."))

    return X, labels.map(TARGET_MAPPING).astype(int), notes


def find_unseen_categories(X, model):
    """Categorical values not seen in training. They are encoded as all zeros (handle_unknown="ignore")."""
    encoder = model.named_steps["preprocessor"].named_steps["encode"].named_transformers_["cat"]
    cleaned = clean_features(X)
    unseen = {}
    for col, known in zip(CATEGORICAL_FEATURES, encoder.categories_):
        mask = ~cleaned[col].isin(known)
        if mask.any():
            unseen[col] = (int(mask.sum()), sorted(cleaned.loc[mask, col].unique())[:5])
    return unseen


def evaluate_all(models, X, y):
    """Predict with every loaded model; return comparison table and per-model details."""
    details, rows = {}, []
    for key, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # e.g. undefined-metric warnings on single-class uploads
            y_pred = model.predict(X)
            y_proba = positive_class_proba(model, X)
            metrics = compute_metrics(y, y_pred, y_proba)
            report = classification_report_frame(y, y_pred)
        details[key] = {"y_pred": y_pred, "y_proba": y_proba, "metrics": metrics,
                        "confusion": confusion_matrix_frame(y, y_pred), "report": report}
        rows.append({"Model": MODEL_NAMES[key], **metrics})
    comparison = pd.DataFrame(rows).set_index("Model")[METRIC_COLUMNS]
    return comparison, details


def rank_models(comparison):
    """Rank 1..n by MCC with AUC as tie-breaker (same rule as Phase 3); NaN AUC sorts last."""
    order = comparison.sort_values(RANKING_METRICS, ascending=False, na_position="last").index
    return pd.Series(range(1, len(order) + 1), index=order).reindex(comparison.index)


def best_model_name(comparison):
    return rank_models(comparison).idxmin()


def predictions_table(df, y, details):
    """Row-level actual vs predicted labels and churn probabilities for every model."""
    table = pd.DataFrame({"row": np.arange(1, len(y) + 1)})
    if ID_COLUMN in df.columns:
        table[ID_COLUMN] = df[ID_COLUMN].to_numpy()
    table["actual_churn"] = y.map(LABEL_NAMES).to_numpy()
    for key, detail in details.items():
        table[f"{key}_predicted"] = pd.Series(detail["y_pred"]).map(LABEL_NAMES).to_numpy()
        table[f"{key}_churn_probability"] = np.round(detail["y_proba"], 6)
    return table


def evaluate_csv(data, filename, models):
    """Full upload flow: read -> validate -> evaluate all models. Returns a result dict."""
    df, notes = read_csv_bytes(data, filename)
    X, y, validation_notes = validate_dataset(df)
    notes += validation_notes
    unseen = find_unseen_categories(X, next(iter(models.values())))
    for col, (count, examples) in unseen.items():
        shown = ", ".join("(blank)" if v in ("", "nan") else repr(v) for v in examples)
        notes.append(("warning", f"{col}: {count} row(s) with categories not seen in training "
                                 f"({shown}); they are encoded as all zeros."))
    if y.nunique() < 2:
        notes.append(("warning", f"Only one Churn class ('{LABEL_NAMES[int(y.iloc[0])]}') is present: "
                                 "AUC is undefined and MCC is 0 by definition."))
    comparison, details = evaluate_all(models, X, y)
    return {"filename": filename, "n_rows": len(df), "y": y, "notes": notes,
            "comparison": comparison, "details": details, "best": best_model_name(comparison),
            "predictions": predictions_table(df, y, details)}


def _row_numbers(mask, limit=5):
    rows = (np.flatnonzero(mask.to_numpy()) + 1)[:limit]
    return ", ".join(map(str, rows)) + (" …" if mask.sum() > limit else "")
