"""
dashboard.py
────────────
Multi-page Streamlit dashboard for HR Analytics.

Pages:
  1. Overview     — headcount, attrition snapshot, KPI cards
  2. Workforce    — demographics, tenure, salary distribution
  3. Risk Monitor — high-risk employees, burnout map
  4. Prediction   — real-time attrition risk estimation

Run: streamlit run src/visualization/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make sure src/ is in path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.kpi_calculator import compute_full_report, top_attrition_risk_employees
from src.etl.extract import generate_sample_data
from src.etl.transform import run_transformation

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HR Analytics · Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e2a3a 0%, #2d3f55 100%);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #4fc3f7;
    }
    .risk-high   { color: #ef5350; font-weight: bold; }
    .risk-medium { color: #ffa726; font-weight: bold; }
    .risk-low    { color: #66bb6a; font-weight: bold; }
    h1 { color: #4fc3f7 !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading HR data…")
def load_data() -> pd.DataFrame:
    processed = Path("data/processed/employees_clean.csv")
    if processed.exists():
        return pd.read_csv(processed)
    raw = generate_sample_data(n_employees=500)
    return run_transformation(raw)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — global filters
# ──────────────────────────────────────────────────────────────────────────────

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.image("https://img.icons8.com/fluency/96/brain.png", width=60)
    st.sidebar.title("HR Analytics")
    st.sidebar.markdown("---")

    st.sidebar.subheader("🔍 Filters")
    departments = ["All"] + sorted(df["department"].unique().tolist())
    selected_dept = st.sidebar.selectbox("Department", departments)

    job_levels = ["All"] + sorted(df["job_level"].unique().tolist())
    selected_level = st.sidebar.selectbox("Job Level", job_levels)

    gender_options = ["All"] + sorted(df["gender"].unique().tolist())
    selected_gender = st.sidebar.selectbox("Gender", gender_options)

    st.sidebar.markdown("---")
    st.sidebar.caption("HR Analytics Pipeline v1.0 · MIT License")

    # Apply filters
    filtered = df.copy()
    if selected_dept != "All":
        filtered = filtered[filtered["department"] == selected_dept]
    if selected_level != "All":
        filtered = filtered[filtered["job_level"] == selected_level]
    if selected_gender != "All":
        filtered = filtered[filtered["gender"] == selected_gender]

    return filtered


# ──────────────────────────────────────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────────────────────────────────────

def page_overview(df: pd.DataFrame) -> None:
    st.title("🧠 HR Analytics — Overview")
    report = compute_full_report(df)

    # KPI cards row 1
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Headcount",       f"{report.headcount:,}")
    c2.metric("📉 Attrition Rate",  f"{report.attrition_rate_pct:.1f}%",
              delta=f"{report.attrition_rate_pct - 8:.1f}% vs benchmark")
    c3.metric("💶 Avg Salary",      f"€{report.avg_salary_eur:,.0f}")
    c4.metric("😊 Satisfaction",    f"{report.avg_satisfaction_score:.2f} / 5")

    # KPI cards row 2
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("⚖️  Gender Pay Gap",  f"{report.gender_pay_gap_pct:.1f}%")
    c6.metric("🔥 Burnout Risk",     f"{report.burnout_risk_pct:.1f}%")
    c7.metric("📅 Absenteeism",      f"{report.absenteeism_rate_pct:.1f}%")
    c8.metric("🎯 eNPS",             f"{report.employee_nps:.0f}")

    st.markdown("---")

    # Attrition by department chart
    dept_data = pd.DataFrame(
        list(report.attrition_by_dept.items()),
        columns=["Department", "Attrition Rate (%)"]
    ).sort_values("Attrition Rate (%)", ascending=True)

    fig = px.bar(
        dept_data,
        x="Attrition Rate (%)",
        y="Department",
        orientation="h",
        color="Attrition Rate (%)",
        color_continuous_scale="RdYlGn_r",
        title="Attrition Rate by Department",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
    )
    st.plotly_chart(fig, use_container_width=True)


def page_workforce(df: pd.DataFrame) -> None:
    st.title("👥 Workforce Analytics")

    col1, col2 = st.columns(2)

    # Age distribution
    fig_age = px.histogram(
        df, x="age", nbins=20, color="gender",
        title="Age Distribution by Gender",
        barmode="overlay", opacity=0.75,
        color_discrete_map={"M": "#4fc3f7", "F": "#f48fb1", "Non-binary": "#a5d6a7"},
    )
    fig_age.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0e0")
    col1.plotly_chart(fig_age, use_container_width=True)

    # Salary by department (box plot)
    fig_salary = px.box(
        df, x="department", y="salary_eur", color="department",
        title="Salary Distribution by Department",
        labels={"salary_eur": "Annual Salary (€)"},
    )
    fig_salary.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0e0")
    col2.plotly_chart(fig_salary, use_container_width=True)

    # Tenure distribution
    fig_tenure = px.histogram(
        df, x="tenure_band", color="job_level",
        title="Headcount by Tenure Band and Job Level",
        category_orders={"tenure_band": ["New (<1y)", "Junior (1-3y)", "Mid (3-7y)", "Senior (7y+)"]},
        barmode="stack",
    )
    fig_tenure.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0e0")
    st.plotly_chart(fig_tenure, use_container_width=True)


def page_risk_monitor(df: pd.DataFrame) -> None:
    st.title("🚨 Risk Monitor")

    high_risk = df[df.get("risk_score", pd.Series([0] * len(df))) > 60] if "risk_score" in df.columns else df.head(0)

    st.subheader(f"⚠️  {len(high_risk)} employees at elevated attrition risk")

    # Risk scatter: satisfaction vs overtime, colored by risk score
    if "risk_score" in df.columns:
        fig = px.scatter(
            df,
            x="satisfaction_score",
            y="overtime_hours_monthly",
            color="risk_score",
            color_continuous_scale="RdYlGn_r",
            hover_data=["employee_id", "department", "job_level"],
            title="Risk Map — Satisfaction vs. Overtime (color = risk score)",
            labels={"satisfaction_score": "Satisfaction (1–5)", "overtime_hours_monthly": "Overtime (h/month)"},
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0e0")
        st.plotly_chart(fig, use_container_width=True)

    # Top risk table
    st.subheader("🎯 Top 10 High-Risk Profiles")
    top_risk = top_attrition_risk_employees(df, top_n=10)
    st.dataframe(top_risk, use_container_width=True)


def page_prediction(df: pd.DataFrame) -> None:
    st.title("🤖 Attrition Risk Estimator")
    st.markdown("Fill in the employee profile below to get a real-time attrition risk estimate.")

    col1, col2, col3 = st.columns(3)

    age = col1.slider("Age", 20, 65, 32)
    tenure = col1.slider("Tenure (months)", 0, 240, 24)
    salary = col2.number_input("Salary (€)", 25000, 150000, 45000, step=1000)
    overtime = col2.slider("Overtime hours / month", 0, 50, 10)
    satisfaction = col3.slider("Satisfaction score", 1.0, 5.0, 3.5, step=0.1)
    months_since_promo = col3.slider("Months since last promotion", 0, 60, 18)

    risk_score = round(
        (overtime / 50 * 30)
        + ((5 - satisfaction) / 4 * 30)
        + (months_since_promo / 60 * 20)
        + ((240 - min(tenure, 240)) / 240 * 20),
        1
    )

    st.markdown("---")

    color = "#ef5350" if risk_score > 65 else "#ffa726" if risk_score > 35 else "#66bb6a"
    label = "🔴 High Risk" if risk_score > 65 else "🟡 Medium Risk" if risk_score > 35 else "🟢 Low Risk"

    st.markdown(f"### Risk Score: **{risk_score:.0f} / 100** — <span style='color:{color}'>{label}</span>",
                unsafe_allow_html=True)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 35], "color": "#1b5e20"},
                {"range": [35, 65], "color": "#e65100"},
                {"range": [65, 100], "color": "#b71c1c"},
            ],
        },
        title={"text": "Attrition Risk Score"},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0e0", height=300)
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    df_full = load_data()
    df = render_sidebar(df_full)

    page = st.sidebar.radio(
        "Navigation",
        ["📊 Overview", "👥 Workforce", "🚨 Risk Monitor", "🤖 Prediction"],
    )

    if page == "📊 Overview":
        page_overview(df)
    elif page == "👥 Workforce":
        page_workforce(df)
    elif page == "🚨 Risk Monitor":
        page_risk_monitor(df)
    elif page == "🤖 Prediction":
        page_prediction(df)


if __name__ == "__main__":
    main()
