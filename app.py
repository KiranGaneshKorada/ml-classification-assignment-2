"""Telco Customer Churn Prediction - Streamlit dashboard (Phase 4).

Run from the project folder:  streamlit run app.py

Evaluates the five saved pipelines in model/ on a labelled test CSV. Models are
loaded once (st.cache_resource) and only used for predict()/predict_proba();
nothing is refitted on uploaded data.
"""
import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.app_logic import (
    MODEL_CONFIGS, MODEL_DESCRIPTIONS, DataValidationError, evaluate_csv, load_models,
    rank_models,
)
from src.evaluate_models import METRIC_COLUMNS, classification_report_text, plot_confusion_matrix
from src.preprocessing import FEATURE_COLUMNS, TARGET, TEST_DATA_PATH
from src.train_models import MODEL_NAMES

st.set_page_config(page_title="Telco Customer Churn Prediction", page_icon="📡",
                   layout="wide", initial_sidebar_state="expanded")

KEY_BY_NAME = {name: key for key, name in MODEL_NAMES.items()}
STAYED, CHURNED, NAVY, BLUE, BLUE_SOFT = "#2a78d6", "#eb6834", "#104281", "#2a78d6", "#a9c9f2"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
METRIC_HELP = {
    "Accuracy": "Share of all records classified correctly.",
    "AUC": "Area under the ROC curve, from predicted churn probabilities. 0.5 = random, 1 = perfect ranking.",
    "Precision": "Of customers predicted to churn, the share who actually churned.",
    "Recall": "Of customers who actually churned, the share the model caught.",
    "F1": "Harmonic mean of precision and recall for the churn class.",
    "MCC": "Matthews correlation coefficient: balanced score using all four confusion-matrix cells "
           "(-1 to 1). Used to pick the best model because the classes are imbalanced.",
}
METRIC_LABELS = {"Accuracy": "Accuracy", "AUC": "AUC (ROC)", "Precision": "Precision",
                 "Recall": "Recall", "F1": "F1 score", "MCC": "MCC"}

st.markdown("""
<style>
.block-container {padding-top: 4rem; padding-bottom: 3rem;}
.hero {background: linear-gradient(120deg, #0d366b 0%, #1c5cab 55%, #2a78d6 100%);
       color: #fff; padding: 26px 32px 22px; border-radius: 16px; margin-bottom: 1.4rem;}
.hero h1 {color: #fff; font-size: 2.05rem; line-height: 1.2; margin: 0 0 4px; padding: 0;}
.hero .sub {color: rgba(255,255,255,.86); font-size: 1.02rem; margin: 0 0 14px;}
.pill {display: inline-block; background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25);
       color: #fff; border-radius: 999px; padding: 3px 12px; margin: 0 6px 6px 0; font-size: .82rem;}
[data-testid="stMetric"] {background: #fff; border: 1px solid rgba(11,11,11,.09); border-radius: 12px;
       padding: 14px 16px 12px; box-shadow: 0 1px 2px rgba(11,11,11,.04);}
[data-testid="stMetricLabel"] p {color: #52514e; font-size: .86rem;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
/* Every delta in this app is informational (share, rank), so hide the up/down arrow. */
[data-testid="stMetricDelta"] svg {display: none;}
.best {border: 1px solid #b7d3f6; border-left: 6px solid #2a78d6; background: #eef5fd;
       border-radius: 12px; padding: 14px 20px; margin: 4px 0 16px;}
.best .label {color: #1c5cab; font-size: .78rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;}
.best .name {font-size: 1.35rem; font-weight: 700; color: #0d366b; margin: 2px 0;}
.best .stats {color: #52514e; font-size: .92rem;}
.step {background: #fff; border: 1px solid rgba(11,11,11,.09); border-radius: 12px; padding: 16px 18px; height: 100%;}
.step .n {color: #2a78d6; font-weight: 700; font-size: .8rem; letter-spacing: .06em;}
.step .t {font-weight: 650; margin: 2px 0 4px;}
.step .d {color: #52514e; font-size: .9rem;}
.model-card {background: #fff; border: 1px solid rgba(11,11,11,.09); border-top: 4px solid #2a78d6;
       border-radius: 12px; padding: 14px 16px; height: 100%;}
.model-card .t {font-weight: 650; margin-bottom: 4px;}
.model-card .d {color: #52514e; font-size: .85rem;}
section[data-testid="stSidebar"] h4 {margin-top: .6rem;}
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
    st.markdown("""
    <div class="hero">
      <h1>📡 Telco Customer Churn Prediction</h1>
      <p class="sub">Machine Learning Classification Dashboard | M.Tech AIML</p>
      <span class="pill">5 trained classifiers</span><span class="pill">6 evaluation metrics</span>
      <span class="pill">Positive class: Churn = Yes</span><span class="pill">Live evaluation on your test CSV</span>
    </div>""", unsafe_allow_html=True)


def render_required_columns():
    cols = pd.DataFrame({"Column": FEATURE_COLUMNS + [TARGET],
                         "Role": ["Input feature"] * len(FEATURE_COLUMNS) + ["Target (Yes / No)"]})
    st.dataframe(cols, hide_index=True, width="stretch", height=280)
    st.caption("customerID is optional and ignored. Column order does not matter. "
               "Blank TotalCharges values are allowed (treated as 0, i.e. not yet billed).")


def render_welcome(models):
    st.info("**Get started:** upload a labelled test CSV in the sidebar, or switch on the bundled "
            "held-out test set, to evaluate all five models.", icon="👈")
    steps = [("STEP 1", "Upload test data", "A CSV with the 19 customer attributes and the actual Churn label."),
             ("STEP 2", "Evaluate all models", "Each saved pipeline preprocesses the raw rows and predicts. "
                                               "No model is retrained."),
             ("STEP 3", "Compare and inspect", "See six metrics side by side, the best model by MCC, confusion "
                                               "matrices, reports and downloadable predictions.")]
    for col, (n, title, desc) in zip(st.columns(3), steps):
        col.markdown(f'<div class="step"><div class="n">{n}</div><div class="t">{title}</div>'
                     f'<div class="d">{desc}</div></div>', unsafe_allow_html=True)

    st.subheader("Available models")
    for col, key in zip(st.columns(len(MODEL_NAMES)), MODEL_NAMES):
        status = "" if key in models else " ⚠️ not loaded"
        col.markdown(f'<div class="model-card"><div class="t">{MODEL_NAMES[key]}{status}</div>'
                     f'<div class="d">{MODEL_DESCRIPTIONS[key]}</div></div>', unsafe_allow_html=True)

    st.subheader("Expected CSV format")
    render_required_columns()


def render_validation_error(exc, filename):
    details = "\n".join(f"- {p}" for p in exc.problems)
    st.error(f"**Could not evaluate `{filename}`: {exc.title}**\n\n{details}", icon="🚫")
    with st.expander("Expected CSV format", expanded=True):
        render_required_columns()


def render_overview(result):
    y = result["y"]
    n, churned = len(y), int(y.sum())
    st.subheader("Dataset overview")
    left, right = st.columns([3, 2], gap="large")
    with left:
        a, b = st.columns(2)
        a.metric("Uploaded file", result["filename"])
        b.metric("Records", f"{n:,}")
        c, d = st.columns(2)
        c.metric("Churn = Yes", f"{churned:,}", f"{churned / n:.1%} of records", delta_color="off")
        d.metric("Churn = No", f"{n - churned:,}", f"{(n - churned) / n:.1%} of records", delta_color="off")
    with right:
        dist = pd.DataFrame({"Churn": ["No (stayed)", "Yes (churned)"], "Customers": [n - churned, churned]})
        dist["Share"] = dist["Customers"] / n
        dist["Label"] = [f"{c:,} ({s:.1%})" for c, s in zip(dist["Customers"], dist["Share"])]
        base = alt.Chart(dist, title="Actual churn distribution").encode(
            y=alt.Y("Churn:N", title=None, sort=None, scale=alt.Scale(paddingInner=0.35)),
            x=alt.X("Customers:Q", title="Customers", scale=alt.Scale(domain=[0, max(dist["Customers"]) * 1.3])),
            tooltip=["Churn", alt.Tooltip("Customers:Q", format=","), alt.Tooltip("Share:Q", format=".1%")])
        bars = base.mark_bar(cornerRadiusEnd=4).encode(
            color=alt.Color("Churn:N", scale=alt.Scale(range=[STAYED, CHURNED]), legend=None))
        labels = base.mark_text(align="left", dx=6, color=INK_2).encode(text="Label:N")
        st.altair_chart((bars + labels).properties(height=170).configure_axis(gridColor=GRID)
                        .configure_view(strokeWidth=0), use_container_width=True)
    for level, message in result["notes"]:
        (st.warning if level == "warning" else st.info)(message, icon="⚠️" if level == "warning" else "ℹ️")


def render_comparison(result, selected_name):
    comparison, best = result["comparison"], result["best"]
    ranks = rank_models(comparison)
    row = comparison.loc[best]
    st.markdown(f"""
    <div class="best"><div class="label">🏆 Best model on this data · ranked by MCC</div>
      <div class="name">{best}</div>
      <div class="stats">MCC {fmt(row['MCC'])} · AUC {fmt(row['AUC'])} · F1 {fmt(row['F1'])} ·
        Accuracy {fmt(row['Accuracy'])} · Precision {fmt(row['Precision'])} · Recall {fmt(row['Recall'])}</div>
    </div>""", unsafe_allow_html=True)

    table = comparison.copy()
    table.insert(0, "Rank (MCC)", ranks)
    styled = (table.style
              .format({m: fmt for m in METRIC_COLUMNS})
              .apply(lambda r: ["background-color: #e3eefc; font-weight: 600" if r.name == best else ""
                                for _ in r], axis=1)
              .highlight_max(subset=METRIC_COLUMNS, axis=0, props="color: #104281; font-weight: 700;"))
    st.dataframe(styled, width="stretch")
    st.caption("All values are calculated live from the uploaded file. Highlighted row = best model by MCC "
               "(ties broken by AUC); bold blue = best value in each column.")

    metric = st.radio("Chart metric", METRIC_COLUMNS, index=METRIC_COLUMNS.index("MCC"),
                      horizontal=True, format_func=lambda m: METRIC_LABELS[m])
    chart_df = comparison.reset_index().dropna(subset=[metric])
    if chart_df.empty:
        st.info(f"{METRIC_LABELS[metric]} is undefined for this data, so it cannot be charted.")
        return
    chart_df["Selected"] = chart_df["Model"].eq(selected_name)
    chart_df["Value"] = chart_df[metric].map(fmt)
    lower = min(0.0, float(chart_df[metric].min()))
    base = alt.Chart(chart_df, title=f"{METRIC_LABELS[metric]} by model").encode(
        y=alt.Y("Model:N", sort="-x", title=None, scale=alt.Scale(paddingInner=0.3),
                axis=alt.Axis(labelLimit=240)),
        x=alt.X(f"{metric}:Q", title=METRIC_LABELS[metric], scale=alt.Scale(domain=[lower, 1])),
        tooltip=["Model"] + [alt.Tooltip(f"{m}:Q", format=".4f") for m in METRIC_COLUMNS])
    bars = base.mark_bar(cornerRadiusEnd=4).encode(
        color=alt.condition(alt.datum.Selected, alt.value(NAVY), alt.value(BLUE_SOFT)))
    labels = base.mark_text(align="left", dx=6, color=INK_2).encode(text="Value:N")
    st.altair_chart((bars + labels).properties(height=250).configure_axis(gridColor=GRID)
                    .configure_view(strokeWidth=0), use_container_width=True)
    st.caption(f"Dark bar = model selected in the sidebar ({selected_name}). Hover a bar for all six metrics.")


def render_model_detail(result, key):
    name = MODEL_NAMES[key]
    detail, comparison = result["details"][key], result["comparison"]
    st.markdown(f"### {name}")
    st.markdown(MODEL_DESCRIPTIONS[key])
    st.code(MODEL_CONFIGS[key], language="python")

    cards = st.columns(6)
    for col, metric in zip(cards, METRIC_COLUMNS):
        value = detail["metrics"][metric]
        rank = comparison[metric].rank(ascending=False, method="min")[name]
        col.metric(METRIC_LABELS[metric], fmt(value),
                   None if pd.isna(value) else f"Rank {int(rank)} of {len(comparison)}",
                   delta_color="off", help=METRIC_HELP[metric])

    cm = detail["confusion"]
    tn, fp, fn, tp = cm.to_numpy().ravel()
    n = int(cm.to_numpy().sum())
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown("#### Confusion matrix")
        fig = plot_confusion_matrix(cm, f"{name} · {n:,} records")
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with right:
        st.markdown("#### Prediction outcomes")
        a, b = st.columns(2)
        a.metric("Correct predictions", f"{tn + tp:,}", f"{(tn + tp) / n:.1%} of records", delta_color="off")
        b.metric("Incorrect predictions", f"{fp + fn:,}", f"{(fp + fn) / n:.1%} of records", delta_color="off")
        outcomes = pd.DataFrame({
            "Outcome": ["True negatives", "True positives", "False positives", "False negatives"],
            "Meaning": ["Stayed, predicted to stay", "Churned, predicted to churn",
                        "Stayed, but flagged as churn (false alarm)", "Churned, but missed"],
            "Count": [tn, tp, fp, fn]})
        st.dataframe(outcomes, hide_index=True, width="stretch")

    st.markdown("#### Classification report")
    report = detail["report"]
    st.dataframe(report.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}",
                                      "Support": "{:,}"}), width="stretch")
    st.caption(f"Positive class: Churn = Yes. Overall accuracy: {fmt(report.attrs.get('accuracy'))}.")
    with st.expander("Plain-text classification report"):
        st.code(classification_report_text(result["y"], detail["y_pred"]), language=None)


def render_predictions(result):
    table = result["predictions"]
    stem = result["filename"].rsplit(".", 1)[0]
    st.markdown("Actual churn label alongside each model's predicted label and churn probability "
                "(probability of Churn = Yes).")
    st.download_button("⬇️ Download predictions for all models (CSV)", table.to_csv(index=False).encode(),
                       file_name=f"churn_predictions_{stem}.csv", mime="text/csv", type="primary")
    agreement = pd.DataFrame({
        "Model": [MODEL_NAMES[k] for k in result["details"]],
        "Predicted churners": [int(d["y_pred"].sum()) for d in result["details"].values()],
        "Actual churners": int(result["y"].sum()),
        "Correct": [int((d["y_pred"] == result["y"].to_numpy()).sum()) for d in result["details"].values()],
    })
    st.dataframe(agreement, hide_index=True, width="stretch")
    st.dataframe(table, hide_index=True, width="stretch", height=380)


# ---------------------------------------------------------------- page flow
render_hero()
models, load_errors = get_models()

with st.sidebar:
    st.markdown("## 📡 Churn Model Lab")
    st.caption("Telco Customer Churn · M.Tech AIML · ML Assignment 2")
    st.markdown("#### 1 · Test data")
    uploaded = st.file_uploader("Upload labelled test data (CSV)", type=["csv"],
                                help="19 input feature columns plus Churn (Yes/No). customerID is optional.")
    use_sample = False
    if TEST_DATA_PATH.exists():
        use_sample = st.toggle("Use bundled held-out test set", disabled=uploaded is not None,
                               help="data/processed/test_data.csv: the 20% stratified test split from training.")
        st.download_button("Download sample test CSV", TEST_DATA_PATH.read_bytes(),
                           file_name="test_data.csv", mime="text/csv", width="stretch")
    st.markdown("#### 2 · Model to inspect")
    selected_name = st.selectbox("Classifier", [MODEL_NAMES[k] for k in models] or ["(no models loaded)"],
                                 disabled=not models)
    st.divider()
    if models:
        st.success(f"{len(models)} of {len(MODEL_NAMES)} models loaded from model/", icon="✅")
    with st.expander("Required CSV columns"):
        st.markdown("\n".join(f"- `{c}`" for c in FEATURE_COLUMNS + [TARGET]))

for message in load_errors:
    st.error(f"Model loading failed: {message}", icon="🚫")
if not models:
    st.error("No trained models could be loaded from the model/ directory, so nothing can be evaluated. "
             "Run `python3 -m src.train_models` to create them.")
    st.stop()

if uploaded is not None:
    csv_bytes, filename = uploaded.getvalue(), uploaded.name
elif use_sample:
    csv_bytes, filename = TEST_DATA_PATH.read_bytes(), TEST_DATA_PATH.name
else:
    render_welcome(models)
    st.stop()

try:
    result = run_evaluation(csv_bytes, filename)
except DataValidationError as exc:
    render_validation_error(exc, filename)
    st.stop()
except Exception as exc:  # noqa: BLE001 - show any unexpected failure instead of a blank page
    st.error(f"Unexpected error while evaluating `{filename}`: {type(exc).__name__}: {exc}", icon="🚫")
    st.stop()

render_overview(result)
st.divider()
tab_compare, tab_detail, tab_predictions = st.tabs(
    ["📊  Compare all models", "🔍  Inspect selected model", "⬇️  Predictions"])
with tab_compare:
    render_comparison(result, selected_name)
with tab_detail:
    render_model_detail(result, KEY_BY_NAME[selected_name])
with tab_predictions:
    render_predictions(result)

st.caption("Models were trained once offline on the stratified training split (Phase 3). This app only calls "
           "predict() and predict_proba() on the saved pipelines; nothing is retrained on uploaded data.")
