"""Telco Customer Churn Prediction - Streamlit dashboard.

Run from the project folder:  streamlit run app.py

Evaluates the five saved pipelines in model/ on a labelled test CSV. Models are
loaded once (st.cache_resource) and only used for predict()/predict_proba();
nothing is refitted on uploaded data.
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.app_logic import (
    MODEL_CONFIGS, MODEL_DESCRIPTIONS, DataValidationError, evaluate_csv, load_models,
)
from src.evaluate_models import METRIC_COLUMNS, plot_confusion_matrix
from src.preprocessing import FEATURE_COLUMNS, TARGET, TEST_DATA_PATH
from src.train_models import MODEL_NAMES

APP_TITLE = "Telco Customer Churn Prediction"
ROLL_NUMBER = "2026AJ05797"
st.set_page_config(page_title=APP_TITLE, page_icon="📡", layout="wide", initial_sidebar_state="expanded")

KEY_BY_NAME = {name: key for key, name in MODEL_NAMES.items()}
METRIC_HELP = {
    "Accuracy": "Proportion of all predictions that are correct.",
    "AUC": "Ability to distinguish churners from non-churners across all decision thresholds "
           "(0.5 = random, 1 = perfect).",
    "Precision": "Proportion of predicted churners who actually churned.",
    "Recall": "Proportion of actual churners correctly identified.",
    "F1": "Harmonic mean of precision and recall.",
    "MCC": "Balanced measure of classification quality (-1 to 1), useful with imbalanced classes. "
           "Used to rank the models.",
}
METRIC_LABELS = {"Accuracy": "Accuracy", "AUC": "AUC (ROC)", "Precision": "Precision",
                 "Recall": "Recall", "F1": "F1 score", "MCC": "MCC"}
EVALUATION_NOTE = ("Use unseen labelled test data for a meaningful evaluation. If your uploaded data includes "
                   "records used during model training, the reported metrics may be overly optimistic.")
TABS = ["🔍  Inspect selected model", "📊  Compare all models", "⬇️  Predictions"]

st.markdown("""
<style>
.block-container {padding-top: 4rem; padding-bottom: 3rem;}
.hero {background: linear-gradient(120deg, #0d366b 0%, #1c5cab 55%, #2a78d6 100%);
       color: #fff; padding: 24px 32px; border-radius: 16px; margin-bottom: 1.4rem;
       display: flex; flex-wrap: wrap; align-items: flex-start; justify-content: space-between; gap: 10px 16px;}
.hero h1 {color: #fff; font-size: 2.05rem; line-height: 1.2; margin: 0; padding: 0;}
.hero .roll {color: rgba(255,255,255,.92); background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.28);
       border-radius: 999px; padding: 3px 12px; font-size: .8rem; letter-spacing: .03em; white-space: nowrap;}
[data-testid="stMetric"] {background: #fff; border: 1px solid rgba(11,11,11,.09); border-radius: 12px;
       padding: 14px 16px 12px; box-shadow: 0 1px 2px rgba(11,11,11,.04);}
[data-testid="stMetricLabel"] p {color: #52514e; font-size: .86rem;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
/* Every delta in this app is informational (rank), so hide the up/down arrow. */
[data-testid="stMetricDelta"] svg {display: none;}
.best {border: 1px solid #b7d3f6; border-left: 6px solid #2a78d6; background: #eef5fd;
       border-radius: 12px; padding: 14px 20px; margin: 4px 0 16px;}
.best .label {color: #1c5cab; font-size: .78rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;}
.best .name {font-size: 1.35rem; font-weight: 700; color: #0d366b; margin: 2px 0;}
.best .stats {color: #52514e; font-size: .92rem;}
section[data-testid="stSidebar"] h4 {margin-top: .6rem;}
/* Wide enough for the longest model name in the selector. */
section[data-testid="stSidebar"][aria-expanded="true"] {min-width: 300px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading saved models…")
def get_models():
    return load_models()


@st.cache_data(show_spinner="Evaluating all models on the uploaded data…", max_entries=8)
def run_evaluation(csv_bytes, filename):
    models, _ = get_models()
    return evaluate_csv(csv_bytes, filename, models)


def fmt(value):
    return "N/A" if pd.isna(value) else f"{value:.4f}"


def render_hero():
    st.markdown(f'<div class="hero"><h1>📡 {APP_TITLE}</h1><span class="roll">Roll No. {ROLL_NUMBER}</span></div>',
                unsafe_allow_html=True)


def render_required_columns():
    cols = pd.DataFrame({"Column": FEATURE_COLUMNS + [TARGET],
                         "Role": ["Input feature"] * len(FEATURE_COLUMNS) + ["Target (Yes / No)"]})
    st.dataframe(cols, hide_index=True, width="stretch", height=280)
    st.caption("customerID is optional and ignored. Column order does not matter. "
               "Blank TotalCharges values are allowed (treated as 0, i.e. not yet billed).")


def render_welcome():
    st.info("**Get started:** upload a labelled test CSV in the sidebar, or switch on the bundled held-out "
            "test set. All five saved models are then evaluated on it; none is retrained.", icon="👈")
    st.subheader("Expected CSV format")
    render_required_columns()


def render_validation_error(exc, filename):
    details = "\n".join(f"- {p}" for p in exc.problems)
    st.error(f"**Could not evaluate `{filename}`: {exc.title}**\n\n{details}", icon="🚫")
    with st.expander("Expected CSV format", expanded=True):
        render_required_columns()


def render_overview(result, bundled):
    y = result["y"]
    n, churned = len(y), int(y.sum())
    st.subheader("Dataset overview")
    source = " (bundled held-out test set)" if bundled else ""
    st.caption(f"Evaluated file: `{result['filename']}`{source}")
    a, b, c = st.columns(3)
    a.metric("Records", f"{n:,}")
    b.metric("Churn = Yes", f"{churned:,}")
    c.metric("Churn = No", f"{n - churned:,}")
    for level, message in result["notes"]:
        (st.warning if level == "warning" else st.info)(message, icon="⚠️" if level == "warning" else "ℹ️")


def render_model_detail(result, key):
    name = MODEL_NAMES[key]
    detail, comparison = result["details"][key], result["comparison"]
    st.markdown(f"### {name}")
    st.caption("Choose another model under **2 · Model to inspect** in the sidebar.")
    st.markdown(MODEL_DESCRIPTIONS[key])
    st.code(MODEL_CONFIGS[key], language="python")

    for col, metric in zip(st.columns(6), METRIC_COLUMNS):
        value = detail["metrics"][metric]
        rank = comparison[metric].rank(ascending=False, method="min")[name]
        col.metric(METRIC_LABELS[metric], fmt(value),
                   None if pd.isna(value) else f"Rank {int(rank)} of {len(comparison)}",
                   delta_color="off", help=METRIC_HELP[metric])

    cm = detail["confusion"]
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.markdown("#### Confusion matrix")
        fig = plot_confusion_matrix(cm, f"{name} · {int(cm.to_numpy().sum()):,} records", cell_names=True)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.caption("Rows are the actual class, columns the predicted class.")
    with right:
        st.markdown("#### Classification report")
        report = detail["report"]
        st.dataframe(report.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}",
                                          "Support": "{:,}"}), width="stretch")
        st.caption(f"Positive class: Churn = Yes. Overall accuracy: {fmt(report.attrs.get('accuracy'))}. "
                   "Support = number of records with that actual class.")


def render_comparison(result):
    comparison, best = result["comparison"], result["best"]
    row = comparison.loc[best]
    st.markdown(f"""
    <div class="best"><div class="label">🏆 Best model on this data · ranked by MCC</div>
      <div class="name">{best}</div>
      <div class="stats">MCC {fmt(row['MCC'])} · AUC {fmt(row['AUC'])} · F1 {fmt(row['F1'])} ·
        Accuracy {fmt(row['Accuracy'])} · Precision {fmt(row['Precision'])} · Recall {fmt(row['Recall'])}</div>
    </div>""", unsafe_allow_html=True)

    styled = (comparison.style.format({m: fmt for m in METRIC_COLUMNS})
              .highlight_max(subset=METRIC_COLUMNS, axis=0, props="color: #104281; font-weight: 700;"))
    st.dataframe(styled, width="stretch")
    st.caption("All values are calculated live from the evaluated file. Bold = best value in each column. "
               "MCC ties are broken by AUC.")


def render_predictions(result):
    table = result["predictions"]
    stem = result["filename"].rsplit(".", 1)[0]
    st.markdown("Actual churn label alongside each model's predicted label and churn probability "
                "(probability of Churn = Yes).")
    st.download_button("⬇️ Download predictions for all models (CSV)", table.to_csv(index=False).encode(),
                       file_name=f"churn_predictions_{stem}.csv", mime="text/csv", type="primary")
    counts = ["Predicted churners", "Actual churners", "Correct"]
    summary = pd.DataFrame({
        "Model": [MODEL_NAMES[k] for k in result["details"]],
        "Predicted churners": [int(d["y_pred"].sum()) for d in result["details"].values()],
        "Actual churners": int(result["y"].sum()),
        "Correct": [int((d["y_pred"] == result["y"].to_numpy()).sum()) for d in result["details"].values()],
    })
    st.dataframe(summary.style.format({c: "{:,}" for c in counts}), hide_index=True, width="stretch")
    st.dataframe(table, hide_index=True, width="stretch", height=380)


# ---------------------------------------------------------------- page flow
render_hero()
models, load_errors = get_models()

with st.sidebar:
    st.markdown("#### 1 · Test data")
    uploaded = st.file_uploader("Upload labelled test data (CSV)", type=["csv"],
                                help="19 input feature columns plus Churn (Yes/No). customerID is optional.")
    use_sample = False
    if TEST_DATA_PATH.exists():
        use_sample = st.toggle("Use bundled held-out test set", disabled=uploaded is not None,
                               help="test_data.csv: the 20% stratified test split, "
                                    "never used for training.")
        st.download_button("Download sample test CSV", TEST_DATA_PATH.read_bytes(),
                           file_name="test_data.csv", mime="text/csv", width="stretch")
    st.caption(f"⚠️ {EVALUATION_NOTE}")
    st.markdown("#### 2 · Model to inspect")
    selected_name = st.selectbox("Classifier", [MODEL_NAMES[k] for k in models] or ["(no models loaded)"],
                                 disabled=not models,
                                 help="Shown in the 'Inspect selected model' tab.")

for message in load_errors:
    st.error(f"Model loading failed: {message}", icon="🚫")
if not models:
    st.error("No trained models could be loaded from the model/ directory, so nothing can be evaluated. "
             "Run `python3 -m src.train_models` to create them.")
    st.stop()

if uploaded is not None:
    csv_bytes, filename, bundled = uploaded.getvalue(), uploaded.name, False
elif use_sample:
    csv_bytes, filename, bundled = TEST_DATA_PATH.read_bytes(), TEST_DATA_PATH.name, True
else:
    render_welcome()
    st.stop()

try:
    result = run_evaluation(csv_bytes, filename)
except DataValidationError as exc:
    render_validation_error(exc, filename)
    st.stop()
except Exception as exc:  # noqa: BLE001 - show any unexpected failure instead of a blank page
    st.error(f"Unexpected error while evaluating `{filename}`: {type(exc).__name__}: {exc}", icon="🚫")
    st.stop()

render_overview(result, bundled)
st.divider()
tab_detail, tab_compare, tab_predictions = st.tabs(TABS)
with tab_detail:
    render_model_detail(result, KEY_BY_NAME[selected_name])
with tab_compare:
    render_comparison(result)
with tab_predictions:
    render_predictions(result)

st.caption("Models were trained once offline on the stratified training split. This app only calls "
           "predict() and predict_proba() on the saved pipelines; nothing is retrained on uploaded data.")
