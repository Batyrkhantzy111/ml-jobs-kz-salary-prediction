import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import ast
import joblib
import os
import streamlit.components.v1 as components
import plotly.express as px
from itertools import chain
from collections import Counter
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="IT salaries Kazakhstan",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DATE = "2026-04-14"
LAST_UPDATE  = "april 14, 2026"

VIZ_DIR = "data/vizualizations"

GRADE_MAP = {"Junior": 1, "Middle": 2, "Senior": 3}
GRADE_OPTIONS = ["Junior", "Middle", "Senior"]

EXP_LABELS = {
    "noExperience": "No experience",
    "between1And3": "1-3 years",
    "between3And6": "3-6 years",
    "moreThan6": "6+ years",
}
EXP_REVERSE = {v: k for k, v in EXP_LABELS.items()}
EXP_ORDER = ["No experience", "1-3 years", "3-6 years", "6+ years"]

EMPLOYMENT_LABELS = {
    "full": "Full-time",
    "part": "Part-time",
    "project":"Project-based",
}
EMPLOYMENT_REVERSE = {v: k for k, v in EMPLOYMENT_LABELS.items()}

WF_COLS = ["wf_ON_SITE", "wf_REMOTE", "wf_HYBRID", "wf_FIELD_WORK"]
WF_LABELS = {
    "wf_ON_SITE": "On-site",
    "wf_REMOTE": "Remote",
    "wf_HYBRID": "Hybrid",
    "wf_FIELD_WORK": "Field work",
}

# Skills with >=5% frequency - same threshold as in training notebook
SKILLS_COL = [
    "1C", "AWS", "Bash", "CI/CD", "Confluence", "Docker", "ELK", "Git",
    "Go", "Grafana", "Java", "JavaScript", "Jira", "Kafka", "Kubernetes",
    "Linux", "MySQL", "Oracle", "PostgreSQL", "Prometheus", "Python",
    "REST", "RabbitMQ", "React", "Redis", "SQL", "TypeScript",
]

SKILL_CATEGORY = {
    "Python": "language", "JavaScript": "language", "TypeScript": "language",
    "Java": "language", "Go": "language",
    "React": "framework",
    "AWS": "cloud",
    "PostgreSQL": "db", "MySQL": "db", "Oracle": "db", "Redis": "db", "SQL": "db",
    "Docker": "tool", "Kubernetes": "tool", "Git": "tool", "CI/CD": "tool",
    "Kafka": "tool", "Linux": "tool", "Bash": "tool", "Prometheus": "tool",
    "Grafana": "tool", "ELK": "tool", "REST": "tool", "RabbitMQ": "tool",
    "Jira": "tool", "Confluence": "tool", "1C": "other",
}

CATEGORY_COLOR = {
    "language": "#7357FD",
    "framework": "#02BE9E",
    "cloud": "#FD6363",
    "db": "#F8C85A",
    "tool": "#49BFF6",
    "other": "#AAAAAA",
}

_dark = {"template": "plotly_dark"}

@st.cache_data
def load_data():
    df_vac = pd.read_csv(f"data/processed/vacancies_clean_{DATA_DATE}.csv")
    df_feat = pd.read_csv(f"data/processed/features_nlp_{DATA_DATE}.csv")

    df_vac["skills_extracted"] = df_vac["skills_extracted"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    # Attached vacancies for further dashboard
    df = pd.merge(df_vac, df_feat[["id", "grade"]], on="id", how="left")
    return df, df_feat


@st.cache_resource
def load_model():
    return joblib.load("models/salary_model.pkl")

def predict_salary(model, df_feat_raw, area, experience, grade_str, employment, wf_selected, skills_selected):
    df_f = df_feat_raw.copy()

    df_f["skills_count"] = df_f[SKILLS_COL].sum(axis=1)
    df_f["grade"] = df_f["grade"].map(GRADE_MAP).fillna(0).astype(int)

    feature_col = [c for c in df_f.columns if c not in {"id", "salary_mid"}]

    row = {col: 0 for col in feature_col}
    row["area"] = area
    row["employment"] = employment
    row["experience"] = experience
    row["has_usd_in_description"] = False
    row["grade"] = GRADE_MAP.get(grade_str, 1)

    for wf_col in WF_COLS:
        row[wf_col] = 1 if WF_LABELS[wf_col] in wf_selected else 0

    for skill in skills_selected:
        if skill in row:
            row[skill] = 1

    row["skills_count"] = sum(1 for s in SKILLS_COL if row.get(s, 0) == 1)

    X = pd.DataFrame([row])[feature_col]
    return float(model.predict(X)[0])

def embed_html(path, height: int = 520):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            components.html(f.read(), height=height, scrolling=False)
    else:
        st.warning(f"Chart not found: {path}")

def page_dashboard(df):
    st.title("Market dashboard")
    st.caption(
        f"Source: hh.kz   |   Last updated: {LAST_UPDATE}   |   {len(df)} vacancies total"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        roles = st.multiselect("Specialty", sorted(df["role"].dropna().unique()), placeholder="All specialties")
    with c2:
        cities = st.multiselect("City", sorted(df["area"].dropna().unique()), placeholder="All cities")
    with c3:
        grades = st.multiselect("Grade", GRADE_OPTIONS, placeholder="All grades")

    dff = df.copy()
    if roles:
        dff = dff[dff["role"].isin(roles)]
    if cities:
        dff = dff[dff["area"].isin(cities)]
    if grades:
        dff = dff[dff["grade"].isin(grades)]

    if dff.empty:
        st.warning("No data for the selected filters")
        return

    dff_labeled = dff[dff["salary_mid"].notna()].copy()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total vacancies", f"{len(dff)}")
    k2.metric("With salary listed", f"{len(dff_labeled)}")
    if len(dff_labeled):
        k3.metric("Median salary", f"{dff_labeled['salary_mid'].median()/1_000:.0f}K KZT")
        k4.metric("Mean salary", f"{dff_labeled['salary_mid'].mean()/1_000:.0f}K KZT")

    st.divider()
    active = (roles or cities or grades)
    if not active:
        # No filters active, just show pre-saved viz
        st.subheader("Top-20 in-demand skills")
        embed_html(os.path.join(VIZ_DIR, "viz_1.html"), height=540)

        st.subheader("Skill demand by specialization")
        embed_html(os.path.join(VIZ_DIR, "viz_2.html"), height=540)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Vacancies by city")
            embed_html(os.path.join(VIZ_DIR, "viz_3.html"), height=480)
        with c2:
            st.subheader("Employment type and work format")
            embed_html(os.path.join(VIZ_DIR, "viz_4.html"), height=480)

        col_c, col_d = st.columns(2)
        with col_c:
            st.subheader("Salary distribution")
            embed_html(os.path.join(VIZ_DIR, "viz_5.html"), height=480)
        with col_d:
            st.subheader("Salary by experience")
            embed_html(os.path.join(VIZ_DIR, "viz_6.html"), height=480)

    else:
        # Filters active: re-do the same charts with filters
        st.info(
            f"Filtered view: {len(dff)} vacancies "
            f"({len(dff_labeled)} with salary listed)"
        )

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Salary distribution")
            if len(dff_labeled):
                sal = dff_labeled["salary_mid"]
                p5, q1, med, q3, p95 = sal.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
                fig = px.histogram(
                    dff_labeled, x="salary_mid", nbins=40,
                    labels={"salary_mid": "Salary (KZT)"},
                    **_dark,
                )
                fig.add_vline(x=p5,  line_dash="dash", line_color="#FF6B6B", line_width=1.5,
                            annotation_text=f"P5 {p5/1_000:.0f}k",  annotation_position="top", annotation_font_size=9)
                fig.add_vline(x=q1,  line_dash="dash", line_color="#FFD166", line_width=1.5,
                            annotation_text=f"P25 {q1/1_000:.0f}k", annotation_position="top", annotation_font_size=9)
                fig.add_vline(x=med, line_dash="dash", line_color="#00C9A7", line_width=1.5,
                            annotation_text=f"Median {med/1_000:.0f}k", annotation_position="top", annotation_font_size=9)
                fig.add_vline(x=q3,  line_dash="dash", line_color="#FFD166", line_width=1.5,
                            annotation_text=f"P75 {q3/1_000:.0f}k", annotation_position="top", annotation_font_size=9)
                fig.add_vline(x=p95, line_dash="dash", line_color="#FF6B6B", line_width=1.5,
                            annotation_text=f"P95 {p95/1_000:.0f}k", annotation_position="top", annotation_font_size=9)
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, width="stretch")

        with c2:
            st.subheader("Salary by experience")
            if len(dff_labeled):
                dff_exp = dff_labeled.copy()
                dff_exp["exp_label"] = dff_exp["experience"].map(EXP_LABELS).fillna("Unknown")
                dff_exp = dff_exp[dff_exp["exp_label"].isin(EXP_ORDER)]
                fig = px.box(
                    dff_exp, x="exp_label", y="salary_mid",
                    category_orders={"exp_label": EXP_ORDER},
                    points="outliers",
                    labels={"exp_label": "Experience", "salary_mid": "Salary (KZT)"},
                    **_dark,
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, width="stretch")

        st.subheader("Top-20 In-Demand Skills")
        all_skills_flat = list(chain.from_iterable(dff["skills_extracted"]))
        if all_skills_flat:
            counts = Counter(all_skills_flat)
            top20 = pd.DataFrame(counts.most_common(20), columns=["skill", "count"])
            top20["category"] = top20["skill"].map(SKILL_CATEGORY).fillna("other")
            top20 = top20.sort_values("count")
            fig = px.bar(
                top20, x="count", y="skill", orientation="h",
                color="category", color_discrete_map=CATEGORY_COLOR,
                labels={"count": "Vacancy count", "skill": ""},
                **_dark,
            )
            fig.update_layout(
                legend_title_text="Category",
                yaxis={"categoryorder": "total ascending"},
                height=520,
            )
            st.plotly_chart(fig, width="stretch")

        st.subheader("Vacancies by city")
        city_counts = dff["area"].value_counts().head(15).reset_index()
        city_counts.columns = ["city", "count"]
        fig = px.bar(
            city_counts, x="city", y="count",
            labels={"count": "Vacancy count", "city": "City"},
            color="count", color_continuous_scale="Blues",
            **_dark,
        )
        fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-20, height=400)
        st.plotly_chart(fig, width="stretch")

def page_calculator(df: pd.DataFrame, df_feat: pd.DataFrame, model):
    st.title("Salary Calculator")
    st.caption(
        f"XGBoost model trained on {df['salary_mid'].notna().sum()} vacancies "
        f"with disclosed salary   |   data as of {LAST_UPDATE}"
    )

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("Parameters")

        specialty = st.selectbox(
            "Specialty",
            options=sorted(df["role"].dropna().unique()),
        )

        c1, c2 = st.columns(2)
        with c1:
            grade = st.selectbox("Grade", GRADE_OPTIONS, index=1)
        with c2:
            experience_label = st.selectbox("Experience", EXP_ORDER, index=1)
        experience = EXP_REVERSE[experience_label]

        c3, c4 = st.columns(2)
        with c3:
            city = st.selectbox("City", sorted(df["area"].dropna().unique()))
        with c4:
            employment_label = st.selectbox("Employment", list(EMPLOYMENT_LABELS.values()))
        employment = EMPLOYMENT_REVERSE[employment_label]

        wf_selected = st.multiselect(
            "Work format", list(WF_LABELS.values()), default=["On-site"]
        )

    with col_right:
        st.subheader("Skills")
        st.caption("Click on the skills you have")

        SKILL_GROUPS = {
            "Languages": ["Python", "Java", "JavaScript", "TypeScript", "Go", "SQL"],
            "Frameworks and tools": ["React", "REST", "Git", "CI/CD", "Linux", "Bash", "1C"],
            "Infrastructure": ["Docker", "Kubernetes", "Kafka", "RabbitMQ", "AWS", "ELK"],
            "Databases": ["PostgreSQL", "MySQL", "Redis", "Oracle"],
            "Monitoring and PM": ["Grafana", "Prometheus", "Confluence", "Jira"],
        }

        selected_skills = []
        for group_name, group_skills in SKILL_GROUPS.items():
            with st.expander(group_name, expanded=True):
                cols = st.columns(3)
                for i, skill in enumerate(group_skills):
                    if skill in SKILLS_COL:
                        if cols[i % 3].checkbox(skill, key=f"skill_{skill}"):
                            selected_skills.append(skill)

    st.divider()
    btn_col, result_col = st.columns([1, 3])
    with btn_col:
        predict_btn = st.button("Calculate salary", type="primary")

    if predict_btn:
        with st.spinner("Running model..."):
            try:
                salary = predict_salary(
                    model, df_feat, city, experience, grade,
                    employment, wf_selected, selected_skills,
                )
                with result_col:
                    st.success(
                        f"Predicted salary: **{salary:,.0f} KZT / month**"
                    )
                    st.caption(
                        f"Specialty: {specialty}  |  {grade}  |  {experience_label}  |  "
                        f"{city}  |  Skills selected: {len(selected_skills)}"
                    )

                st.subheader("Comparison with market benchmarks")
                comp_df = df[df["salary_mid"].notna()].copy()

                rows = [{"Segment": "All vacancies",
                         "Median (KZT)": comp_df["salary_mid"].median()}]

                city_med = comp_df[comp_df["area"] == city]["salary_mid"].median()
                if not np.isnan(city_med):
                    rows.append({"Segment": f"City: {city}", "Median (KZT)": city_med})

                grade_med = comp_df[comp_df["grade"] == grade]["salary_mid"].median()
                if not np.isnan(grade_med):
                    rows.append({"Segment": f"Grade: {grade}", "Median (KZT)": grade_med})

                role_med = comp_df[comp_df["role"] == specialty]["salary_mid"].median()
                if not np.isnan(role_med):
                    rows.append({"Segment": f"Role: {specialty[:35]}", "Median (KZT)": role_med})

                rows.append({"Segment": "Prediction", "Median (KZT)": salary})

                comp = pd.DataFrame(rows)
                comp["Median (KZT)"] = comp["Median (KZT)"].round(0)
                colors = [
                    "#00C9A7" if r == "Prediction" else "#49BFF6"
                    for r in comp["Segment"]
                ]
                fig_cmp = go.Figure(go.Bar(
                    x=comp["Median (KZT)"], y=comp["Segment"],
                    orientation="h",
                    marker_color=colors,
                    text=comp["Median (KZT)"].apply(lambda v: f"{v/1_000:.0f}K KZT"),
                    textposition="outside",
                ))
                fig_cmp.update_layout(
                    xaxis_title="Salary (KZT)",
                    margin=dict(l=200, r=80),
                    height=280,
                    **_dark,
                )
                st.plotly_chart(fig_cmp, width="stretch")

            except (KeyError, ValueError) as e:
                st.error(f"Check your inputs: {e}")

def page_shap():
    st.title("Skills and money")
    st.caption(
        f"SHAP analysis: which skills impact the salary and by how many tenge  \n"
        f"Data {LAST_UPDATE}"
    )

    st.info(
        "How to read: the bar shows by how many KZT the predicted salary shifts on average when a "
        "skill appears in a vacancy, all else being equal"
    )

    embed_html(os.path.join(VIZ_DIR, "viz_8.html"), height=520)
    embed_html(os.path.join(VIZ_DIR, "viz_9.html"), height=520)
    embed_html(os.path.join(VIZ_DIR, "viz_10.html"), height=520)

    st.caption(
        "Keep in mind  \n"
        "SHAP values show how closely a skill is linked to salary in the training data - " 
        "they don't prove that the skill causes a higher salary.  \n"
        "Also the model was trained on a small dataset: only 367 job "
        "postings had a salary listed, so the numbers may not be perfectly accurate"
    )

def main():
    df, df_feat = load_data()
    model = load_model()

    with st.sidebar:
        st.markdown("## IT salaries Kazakhstan")
        st.caption(f"Data updated: **{LAST_UPDATE}**")
        st.divider()
        page = st.radio(
            "Navigation",
            ["Market dashboard", "Salary calculator", "Skills and money"],
        )
        st.divider()
        st.caption("Source: hh.kz  |  model: XGBoost  |  SHAP skill analysis")

    if page == "Market dashboard":
        page_dashboard(df)
    elif page == "Salary calculator":
        page_calculator(df, df_feat, model)
    elif page == "Skills and money":
        page_shap()


if __name__ == "__main__":
    main()
