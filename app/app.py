# Annapurna - College Mess Food-Waste Predictor (Streamlit app)
# Run from the annapurna-ml folder:  streamlit run app/app.py

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

MODEL_PATHS = [
    ROOT / "models" / "annapurna_models.joblib",
    APP_DIR / "annapurna_models.joblib",
    ROOT / "annapurna_models.joblib",
    ROOT / "notebooks" / "annapurna_models.joblib",
]

# Results from the project README.
# Update these numbers after retraining on real mess data.
PERFORMANCE = {
    "Test R² (regression)": "0.77",
    "Test MAE (regression)": "7.2 plates",
    "Validation R² (regression)": "0.74",
    "SVM recall (waste classifier)": "0.66",
    "SVM F1 (waste classifier)": "0.55",
}
DATA_NOTE = (
    "Trained on a small dataset (176 meal records) that is currently "
    "**simulated**. Real mess records will replace it."
)

st.set_page_config(page_title="Annapurna", page_icon="🍛", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1100px;}
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border-radius: 10px;
        padding: 12px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------
@st.cache_resource
def load_bundle():
    for path in MODEL_PATHS:
        if path.exists():
            return joblib.load(path)
    return None


@st.cache_data
def load_menu_map():
    """meal -> menu items seen for that meal in the project data (if available)."""
    data_dir = ROOT / "data"
    if not data_dir.exists():
        return {}
    for path in sorted(data_dir.rglob("*.csv")):
        try:
            cols = pd.read_csv(path, nrows=0).columns
            if {"meal", "menu_main"} <= set(cols):
                df = pd.read_csv(path, usecols=["meal", "menu_main"])
                return {m: sorted(g["menu_main"].unique()) for m, g in df.groupby("meal")}
        except Exception:
            continue
    return {}


def predict_plan(bundle, base, planned, margin):
    """Predicted plates eaten, suggested amount and waste-risk flag for one plan."""
    reg_row = pd.DataFrame([base])[bundle["reg_features"]]
    eaten = float(bundle["regressor"].predict(reg_row)[0])
    eaten = max(0.0, min(eaten, float(base["students_registered"])))

    clf_in = dict(base)
    clf_in["plates_cooked"] = planned
    clf_in["cook_ratio"] = planned / max(1, base["students_registered"])
    clf_row = pd.DataFrame([clf_in])[bundle["clf_features"]]
    flagged = int(bundle["classifier"].predict(clf_row)[0]) == 1

    return {
        "eaten": eaten,
        "suggested": int(round(eaten * (1 + margin / 100))),
        "leftover": planned - eaten,
        "flagged": flagged,
    }


def plan_status(leftover, planned, limit, flagged):
    pct = leftover / max(1, planned)
    if pct > limit:
        return "High waste"
    if leftover < 0:
        return "Short"
    if flagged:
        return "Watch"
    return "OK"


# ---------------------------------------------------------------
# Load models
# ---------------------------------------------------------------
bundle = load_bundle()
if bundle is None:
    st.error(
        "Could not find `annapurna_models.joblib`. Place it in the `models` "
        "folder and refresh the page."
    )
    st.stop()

menu_map = load_menu_map()

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    st.title("🍛 Annapurna")
    st.caption("Mess food-waste predictor")
    st.markdown(
        "Enter tomorrow's meal details and your cooking plan. Annapurna "
        "predicts how many plates will be eaten and warns you when a plan "
        "is likely to waste food."
    )
    st.divider()
    st.subheader("Settings")
    margin = st.slider("Safety margin on suggested amount (%)", 0, 15, 5)
    waste_limit_pct = st.slider("Warn when expected leftover exceeds (%)", 5, 40, 15)
    waste_limit = waste_limit_pct / 100
    st.divider()
    st.caption("Built with scikit-learn and Streamlit.")

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.title("🍛 Annapurna")
st.markdown("**Cook what's needed, waste less.** Demand prediction for a college mess.")

tab_predict, tab_whatif, tab_about = st.tabs(
    ["📊 Predict", "🔍 What-if analysis", "ℹ️ About the model"]
)

# ---------------------------------------------------------------
# Tab 1: Predict
# ---------------------------------------------------------------
with tab_predict:
    st.subheader("Meal details")
    c1, c2, c3 = st.columns(3)
    with c1:
        day = st.selectbox("Day of week", bundle["options"]["day_of_week"])
        meal = st.selectbox("Meal", bundle["options"]["meal"])
    with c2:
        menu_choices = menu_map.get(meal, bundle["options"]["menu_main"])
        menu = st.selectbox("Main menu item", menu_choices)
        registered = st.number_input("Students registered", 1, 2000, 420, 5)
    with c3:
        temperature = st.slider("Temperature (°C)", 10.0, 45.0, 30.0, 0.5)
        exam_week = st.checkbox("Exam week")
        event = st.checkbox("Holiday or college event")
        rain = st.checkbox("Rain expected")

    st.subheader("Cooking plan")
    planned = st.number_input("Plates you plan to cook", 1, 3000, 440, 5)

    submitted = st.button("Predict", type="primary")

    if submitted:
        st.session_state["inputs"] = {
            "base": {
                "day_of_week": day,
                "meal": meal,
                "menu_main": menu,
                "exam_week": int(exam_week),
                "holiday_or_event": int(event),
                "rain": int(rain),
                "temperature_c": temperature,
                "students_registered": registered,
            },
            "planned": int(planned),
        }

    saved = st.session_state.get("inputs")
    if saved is None:
        st.info("Fill in the meal details above and click **Predict**.")
    else:
        base, plan = saved["base"], saved["planned"]
        try:
            res = predict_plan(bundle, base, plan, margin)
        except Exception as err:
            st.error(f"Prediction failed: {err}")
            st.stop()

        st.divider()
        st.subheader("Results")
        st.caption(f"{base['day_of_week']} · {base['meal']} · {base['menu_main']} · "
                   f"{base['students_registered']} students registered")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted plates eaten", f"{res['eaten']:.0f}")
        m2.metric("Suggested to cook", f"{res['suggested']}",
                  delta=f"{res['suggested'] - plan:+d} vs your plan",
                  delta_color="inverse")
        short = res["leftover"] < 0
        m3.metric("Expected shortfall" if short else "Expected leftover",
                  f"{abs(res['leftover']):.0f} plates")
        m4.metric("Shortfall share of plan" if short else "Leftover share of plan",
                  f"{abs(res['leftover']) / plan:.0%}")

        status = plan_status(res["leftover"], plan, waste_limit, res["flagged"])
        if status == "High waste":
            st.warning(
                f"⚠️ **High waste expected:** about {res['leftover']:.0f} plates "
                f"({res['leftover'] / plan:.0%} of your plan) may be left over. "
                f"Consider cooking closer to **{res['suggested']} plates**."
            )
        elif status == "Short":
            st.info(
                "Your plan is below predicted demand, so some students may "
                f"miss out. Consider cooking about **{res['suggested']} plates**."
            )
        elif status == "Watch":
            st.warning(
                "⚠️ The waste classifier flags this plan as risky. "
                f"Consider cooking closer to **{res['suggested']} plates**."
            )
        else:
            st.success("✅ This plan looks reasonable (low waste risk).")

        if plan > 1.25 * base["students_registered"]:
            st.caption(
                "Note: this plan is much larger than the number of registered "
                "students, which is outside what the model usually sees, so "
                "treat the risk flag with extra caution."
            )

        chart = pd.DataFrame(
            {"Plates": [plan, round(res["eaten"]), res["suggested"]]},
            index=["Your plan", "Predicted eaten", "Suggested"],
        )
        st.bar_chart(chart)

# ---------------------------------------------------------------
# Tab 2: What-if analysis
# ---------------------------------------------------------------
with tab_whatif:
    saved = st.session_state.get("inputs")
    if saved is None:
        st.info("Run a prediction in the **Predict** tab first.")
    else:
        base, plan = saved["base"], saved["planned"]
        reg = bundle["regressor"].predict(
            pd.DataFrame([base])[bundle["reg_features"]]
        )[0]
        reg = max(0.0, min(float(reg), float(base["students_registered"])))

        st.subheader("What if you cook a different amount?")
        st.caption(
            "Same meal, different cooking plans. Expected leftover comes from "
            "the regression model, and the risk flag from the SVM classifier."
        )

        reg_n = base["students_registered"]
        plans = sorted({
    max(1, int(reg_n * f))
    for f in (0.7, 0.8, 0.9, 1.0, 1.05, 1.1, 1.2)
} | {
    max(1, plan),
    max(1, int(round(reg * (1 + margin / 100))))
})
        rows = []
        for p in plans:
            r = predict_plan(bundle, base, p, margin)
            rows.append({
                "Plates cooked": p,
                "Expected leftover": round(r["leftover"]),
                "Leftover %": f"{r['leftover'] / max(1, p):.0%}",
                "Status": plan_status(r["leftover"], p, waste_limit, r["flagged"]),
            })
        table = pd.DataFrame(rows)
        st.dataframe(table, hide_index=True)
        st.line_chart(table.set_index("Plates cooked")["Expected leftover"])

# ---------------------------------------------------------------
# Tab 3: About
# ---------------------------------------------------------------
with tab_about:
    st.subheader("How it works")
    st.markdown(
        """
        - **Demand model:** Multiple Linear Regression predicts how many plates
          will be eaten from the day, meal, menu, weather, exam-week/event flags
          and registered students. It never sees the amount cooked, because
          that is the decision it is meant to support.
        - **Waste classifier:** An SVM flags cooking plans that are likely to
          produce high waste.
        - **Warning rule:** The app warns mainly from the regression estimate
          (expected leftover above the limit in the sidebar), with the
          classifier as a second check.
        """
    )

    st.subheader("Model performance")
    cols = st.columns(len(PERFORMANCE))
    for col, (name, value) in zip(cols, PERFORMANCE.items()):
        col.metric(name, value)

    st.subheader("Limitations")
    st.markdown(
        f"""
        - {DATA_NOTE}
        - The test set had only 27 rows, so scores are rough.
        - The waste classifier is modest (no model beat the *always No*
          baseline on accuracy), which is why the leftover estimate drives
          the main warning.
        - Predictions are a guide for the kitchen, not a guarantee.
        """
    )