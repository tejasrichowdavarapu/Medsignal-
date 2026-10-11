"""Streamlit dashboard for the saved MedSignal educational prototype."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import inspect_heart_disease_data, load_heart_disease_data
from src.insights import global_feature_importance, local_feature_contributions
from src.model_artifact import load_model_artifact, predict_with_artifact
from src.preprocessing import FEATURE_COLUMNS

PREDICTION_DEFAULTS = {
    "age": 54,
    "sex": 1,
    "cp": 1,
    "trestbps": 130,
    "chol": 240,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0,
    "thal": 3,
}

st.set_page_config(
    page_title="MedSignal | Heart Health Analytics",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #102a43;
        --teal: #0f8b8d;
        --slate: #526579;
        --canvas: #f4f7fa;
        --line: #dce5ed;
    }
    .stApp { background: var(--canvas); color: var(--navy); }
    [data-testid="stHeader"] { background: rgba(244, 247, 250, .94); }
    [data-testid="stSidebar"] {
        background: #102a43;
        border-right: 1px solid #203e58;
    }
    [data-testid="stSidebar"] * { color: #edf5f8; }
    .brand-mark {
        display: inline-flex; align-items: center; justify-content: center;
        width: 2.6rem; height: 2.6rem; margin-right: .65rem;
        border-radius: .8rem; background: #d8f2f0; color: #087b7d;
        font-size: 1.45rem; vertical-align: middle;
    }
    .hero-title { color: var(--navy); font-size: 2rem; font-weight: 750; margin: 0; }
    .hero-copy { color: var(--slate); margin-top: .35rem; }
    .metric-card {
        background: white; border: 1px solid var(--line); border-radius: 14px;
        padding: 1rem 1.15rem; min-height: 108px;
        box-shadow: 0 2px 10px rgba(16, 42, 67, .035);
    }
    .metric-label { color: var(--slate); font-size: .85rem; }
    .metric-value { color: var(--navy); font-size: 1.7rem; font-weight: 700; }
    .section-card {
        background: white; border: 1px solid var(--line); border-radius: 14px;
        padding: 1rem 1.15rem;
    }
    div.stButton > button, div.stFormSubmitButton > button {
        border-radius: 9px; font-weight: 650;
    }
    div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {
        background: var(--teal); border-color: var(--teal);
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        border-color: var(--teal); color: var(--teal);
    }
    .disclaimer {
        border-left: 4px solid #0f8b8d; background: #e9f5f5; color: #264653;
        padding: .8rem 1rem; border-radius: 0 9px 9px 0; margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the trained model…")
def get_artifact():
    """Load the saved model once per Streamlit server process."""
    return load_model_artifact()


def show_metric(label: str, value: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def show_header() -> None:
    st.markdown(
        '<div><span class="brand-mark">♥</span>'
        '<span class="hero-title">MedSignal</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-copy">A transparent, educational look at a small '
        "heart-disease dataset and a baseline machine-learning model.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="disclaimer"><strong>Educational prototype only.</strong> '
        "This model is not a medical diagnostic tool. Do not use its output to "
        "make health decisions.</div>",
        unsafe_allow_html=True,
    )


def confusion_figure(matrix: list[list[int]]) -> go.Figure:
    return px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale=[[0, "#e7f4f3"], [1, "#0f8b8d"]],
        labels={
            "x": "Predicted label",
            "y": "Actual label",
            "color": "Records",
        },
        x=["No disease", "Disease present"],
        y=["No disease", "Disease present"],
        aspect="auto",
    ).update_layout(
        title="Held-out confusion matrix",
        margin=dict(l=10, r=10, t=55, b=10),
        font=dict(color="#102a43"),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )


def render_overview(artifact: dict) -> None:
    st.subheader("Overview")
    st.write(
        "MedSignal demonstrates a reproducible Logistic Regression workflow using "
        "the UCI Heart Disease dataset. The displayed metrics come from the "
        "saved model's held-out test evaluation."
    )

    try:
        summary = inspect_heart_disease_data(load_heart_disease_data())
    except (RuntimeError, ValueError) as error:
        st.error(f"Could not load the dataset overview: {error}")
        return

    metric_row = st.columns(4)
    metric_row[0].metric("Dataset records", f"{summary['rows']:,}")
    metric_row[1].metric("Input features", str(summary["feature_count"]))
    metric_row[2].metric("Test records", str(artifact["metrics"]["test_samples"]))
    metric_row[3].metric("Model", artifact["metrics"]["model"])

    st.markdown("#### Held-out performance")
    metrics = artifact["metrics"]
    metric_row = st.columns(5)
    for column, key, label in zip(
        metric_row,
        ("accuracy", "precision", "recall", "f1", "roc_auc"),
        ("Accuracy", "Precision", "Recall", "F1 score", "ROC-AUC"),
    ):
        column.metric(label, f"{metrics[key]:.1%}")

    chart_left, chart_right = st.columns([1.1, 1])
    with chart_left:
        st.plotly_chart(
            confusion_figure(metrics["confusion_matrix"]),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )
        st.caption(
            "Rows are actual labels; columns are predictions. Class 0 means no "
            "disease in the dataset; class 1 means disease present."
        )
    with chart_right:
        target_counts = pd.DataFrame(
            [
                {"Source label": str(label), "Records": values["count"]}
                for label, values in summary["target_distribution"].items()
            ]
        )
        distribution_figure = px.bar(
            target_counts,
            x="Source label",
            y="Records",
            color="Source label",
            color_discrete_sequence=["#0f8b8d", "#3c6e71", "#6b8f9c", "#88a9b3", "#b2c6cd"],
        )
        distribution_figure.update_layout(
            title="Dataset target labels",
            showlegend=False,
            margin=dict(l=10, r=10, t=55, b=10),
            font=dict(color="#102a43"),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        st.plotly_chart(
            distribution_figure,
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )
        st.caption(
            "UCI labels 1–4 are combined into “disease present” for model training."
        )

    st.caption(
        "One stratified 80/20 split (random seed 42). The dataset is small; "
        "performance here does not establish clinical validity or generalization."
    )


def reset_form() -> None:
    for key, value in PREDICTION_DEFAULTS.items():
        st.session_state[key] = value
    st.session_state.pop("prediction_result", None)
    st.session_state.pop("prediction_inputs", None)


def render_prediction(artifact: dict) -> None:
    st.subheader("Prediction interface")
    st.write(
        "Enter measurements to run the saved model. These fields should represent "
        "a single row in the UCI dataset's feature schema."
    )
    st.caption(
        "Defaults are form starting values only; they are not a sample patient "
        "or a displayed prediction. Change the fields before submitting."
    )

    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            age = st.number_input(
                "Age (years)", 18, 100, value=PREDICTION_DEFAULTS["age"], key="age"
            )
            sex = st.selectbox(
                "Sex (dataset encoding)",
                options=[0, 1],
                format_func=lambda value: "Female (0)" if value == 0 else "Male (1)",
                key="sex",
            )
            cp = st.selectbox(
                "Chest-pain type",
                options=[1, 2, 3, 4],
                format_func=lambda value: {
                    1: "1 · Typical angina",
                    2: "2 · Atypical angina",
                    3: "3 · Non-anginal pain",
                    4: "4 · Asymptomatic",
                }[value],
                key="cp",
            )
            trestbps = st.number_input(
                "Resting blood pressure (mm Hg)",
                60,
                250,
                value=PREDICTION_DEFAULTS["trestbps"],
                key="trestbps",
            )
            chol = st.number_input(
                "Serum cholesterol (mg/dl)",
                80,
                700,
                value=PREDICTION_DEFAULTS["chol"],
                key="chol",
            )
            fbs = st.selectbox(
                "Fasting blood sugar > 120 mg/dl",
                options=[0, 1],
                format_func=lambda value: "No (0)" if value == 0 else "Yes (1)",
                key="fbs",
            )
            restecg = st.selectbox(
                "Resting ECG result",
                options=[0, 1, 2],
                format_func=lambda value: {
                    0: "0 · Normal",
                    1: "1 · ST-T abnormality",
                    2: "2 · Probable/definite LV hypertrophy",
                }[value],
                key="restecg",
            )
        with right:
            thalach = st.number_input(
                "Maximum heart rate achieved (bpm)",
                50,
                250,
                value=PREDICTION_DEFAULTS["thalach"],
                key="thalach",
            )
            exang = st.selectbox(
                "Exercise-induced angina",
                options=[0, 1],
                format_func=lambda value: "No (0)" if value == 0 else "Yes (1)",
                key="exang",
            )
            oldpeak = st.number_input(
                "ST depression (oldpeak)",
                min_value=0.0,
                max_value=10.0,
                value=PREDICTION_DEFAULTS["oldpeak"],
                step=0.1,
                key="oldpeak",
            )
            slope = st.selectbox(
                "Peak-exercise ST slope",
                options=[1, 2, 3],
                format_func=lambda value: {
                    1: "1 · Upsloping",
                    2: "2 · Flat",
                    3: "3 · Downsloping",
                }[value],
                key="slope",
            )
            ca = st.selectbox(
                "Major vessels colored by fluoroscopy",
                options=[0, 1, 2, 3],
                key="ca",
            )
            thal = st.selectbox(
                "Thalassemia result",
                options=[3, 6, 7],
                format_func=lambda value: {
                    3: "3 · Normal",
                    6: "6 · Fixed defect",
                    7: "7 · Reversible defect",
                }[value],
                key="thal",
            )

        submitted = st.form_submit_button(
            "Run prediction", type="primary", use_container_width=True
        )

    st.button("Reset form", on_click=reset_form)

    if submitted:
        values = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }
        try:
            inputs = pd.DataFrame([values], columns=FEATURE_COLUMNS)
            st.session_state["prediction_result"] = predict_with_artifact(
                artifact, inputs
            )
            st.session_state["prediction_inputs"] = inputs
        except (ValueError, RuntimeError) as error:
            st.error(f"Could not produce a prediction: {error}")

    result = st.session_state.get("prediction_result")
    if result:
        st.markdown("#### Model output")
        probability = result["probability_disease_present"]
        if result["prediction"] == 1:
            st.warning(
                "The model classified this input as **disease present** in the "
                "dataset's binary target."
            )
        else:
            st.success(
                "The model classified this input as **no disease** in the "
                "dataset's binary target."
            )
        st.metric("Model-estimated probability of class 1", f"{probability:.1%}")
        st.progress(probability)
        st.caption(
            "This is the trained model's estimated class probability, not a "
            "diagnosis or a measure of certainty about a person's health."
        )


def render_insights(artifact: dict) -> None:
    st.subheader("Model insights")
    st.write(
        "Logistic Regression estimates a weighted relationship between encoded "
        "features and the two dataset labels. Continuous inputs are standardized; "
        "categorical inputs are one-hot encoded."
    )

    left, right = st.columns([1, 1])
    metrics = artifact["metrics"]
    with left:
        st.plotly_chart(
            confusion_figure(metrics["confusion_matrix"]),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )
        st.markdown("#### Held-out metrics")
        columns = st.columns(3)
        for column, key, label in zip(
            columns,
            ("accuracy", "precision", "recall"),
            ("Accuracy", "Precision", "Recall"),
        ):
            column.metric(label, f"{metrics[key]:.1%}")
        columns = st.columns(2)
        columns[0].metric("F1 score", f"{metrics['f1']:.1%}")
        columns[1].metric("ROC-AUC", f"{metrics['roc_auc']:.1%}")

    with right:
        try:
            importance = global_feature_importance(artifact["model"])
        except (ValueError, RuntimeError) as error:
            st.error(f"Could not calculate model coefficient insights: {error}")
        else:
            figure = px.bar(
                importance.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                labels={
                    "importance": "Coefficient magnitude",
                    "feature": "Input feature",
                },
                color_discrete_sequence=["#0f8b8d"],
            )
            figure.update_layout(
                title="Relative coefficient magnitude",
                margin=dict(l=10, r=10, t=55, b=10),
                font=dict(color="#102a43"),
                paper_bgcolor="white",
                plot_bgcolor="white",
            )
            st.plotly_chart(
                figure,
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True},
            )
            st.caption(
                "Magnitude combines one-hot category coefficients per source "
                "feature. It is a model-specific association, not causation or "
                "proof of clinical importance."
            )

    if st.session_state.get("prediction_result") is not None:
        st.markdown("#### Contribution for the latest submitted input")
        if "prediction_inputs" in st.session_state:
            try:
                contributions = local_feature_contributions(
                    artifact["model"], st.session_state["prediction_inputs"]
                )
                contributions = contributions.reindex(
                    contributions["contribution"].abs().sort_values().index
                )
                figure = px.bar(
                    contributions,
                    x="contribution",
                    y="feature",
                    orientation="h",
                    color="contribution",
                    color_continuous_scale=["#64748b", "#dce5ed", "#0f8b8d"],
                    labels={
                        "contribution": "Contribution to log-odds",
                        "feature": "Input feature",
                    },
                )
                figure.update_layout(
                    margin=dict(l=10, r=10, t=20, b=10),
                    font=dict(color="#102a43"),
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    coloraxis_showscale=False,
                )
                st.plotly_chart(
                    figure,
                    use_container_width=True,
                    config={"displayModeBar": False, "responsive": True},
                )
                st.caption(
                    "Positive contributions move the model's score toward class 1; "
                    "negative contributions move it toward class 0. These are "
                    "associations learned by this model, not causal explanations."
                )
            except (ValueError, RuntimeError) as error:
                st.error(f"Could not calculate this input's contribution: {error}")

    with st.expander("Limitations and intended use"):
        st.markdown(
            """
            - This is an educational prototype, not a medical device or diagnostic tool.
            - The dataset is small (303 records) and represents a limited source
              population; performance may not generalize to other people or settings.
            - The target represents the dataset's angiographic labels, not a
              complete clinical assessment.
            - A single held-out split gives an uncertain estimate of performance.
              These results do not establish safety, fairness, or clinical validity.
            - Probabilities and feature contributions are model outputs, not
              medical advice, causal evidence, or certainty about an individual.
            """
        )


show_header()

try:
    model_artifact = get_artifact()
except (FileNotFoundError, RuntimeError) as error:
    st.error(f"Saved model is unavailable: {error}")
    st.code(r".\.venv\Scripts\python.exe train_model.py", language="powershell")
    st.stop()

st.sidebar.markdown("## MedSignal")
st.sidebar.caption("Educational ML prototype")
page = st.sidebar.radio(
    "Application sections",
    ["Overview", "Prediction", "Model insights"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Baseline: {model_artifact['metrics']['model']} · "
    f"seed {model_artifact['metrics']['random_seed']}"
)
st.sidebar.caption("For educational exploration only.")

if page == "Overview":
    render_overview(model_artifact)
elif page == "Prediction":
    render_prediction(model_artifact)
else:
    render_insights(model_artifact)
