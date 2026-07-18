import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


st.set_page_config(
    page_title="Cybersecurity Incident Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIRECTORY,
    "data",
    "clustered_incidents.csv",
)
API_URL = os.getenv(
    "API_URL",
    "https://cybersecurity-resolution-api.onrender.com",
)

NAVY = "#071A2F"
BLUE = "#1473E6"
CYAN = "#22C7F2"
TEAL = "#12B8A6"
ORANGE = "#FF9F43"
PURPLE = "#7C5CFC"
PALETTE = [BLUE, CYAN, TEAL, ORANGE, PURPLE, "#EF5DA8"]


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #F5F8FC;
        color: #15263A;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071A2F 0%, #0B2947 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] * {
        color: #EAF4FF;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 0.55rem 0.65rem;
        margin: 0.15rem 0;
        border-radius: 0.65rem;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(34, 199, 242, 0.12);
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 1.25rem;
        margin-bottom: 1.35rem;
        color: white;
        background:
            radial-gradient(circle at 88% 15%, rgba(34,199,242,.28), transparent 28%),
            linear-gradient(135deg, #071A2F 0%, #0C3D68 62%, #0C6F91 100%);
        box-shadow: 0 18px 45px rgba(7, 26, 47, 0.16);
    }

    .hero-kicker {
        color: #67DCF7;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }

    .hero h1 {
        color: white;
        font-size: clamp(2rem, 4vw, 3.25rem);
        line-height: 1.08;
        margin: 0 0 0.65rem 0;
    }

    .hero p {
        color: #D7E9F7;
        max-width: 760px;
        font-size: 1rem;
        margin: 0;
    }

    .section-title {
        margin: 1.25rem 0 0.2rem;
        font-size: 1.55rem;
        font-weight: 700;
        color: #10283F;
    }

    .section-copy {
        margin: 0 0 1.1rem;
        color: #64788C;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E1EAF3;
        border-radius: 1rem;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 22px rgba(18, 50, 79, 0.06);
    }

    div[data-testid="stMetric"] label {
        color: #63778A;
        font-weight: 600;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0B3558;
        font-weight: 700;
    }

    div[data-testid="stPlotlyChart"],
    div[data-testid="stDataFrame"],
    div[data-testid="stForm"] {
        background: white;
        border: 1px solid #E1EAF3;
        border-radius: 1rem;
        padding: 0.6rem;
        box-shadow: 0 8px 22px rgba(18, 50, 79, 0.05);
    }

    div[data-testid="stForm"] {
        padding: 1.25rem;
    }

    .model-note {
        background: #EDF7FF;
        border-left: 4px solid #1473E6;
        border-radius: 0.8rem;
        padding: 1rem 1.1rem;
        color: #294A65;
        margin-top: 1rem;
    }

    .prediction-result {
        background: linear-gradient(135deg, #071A2F 0%, #0D5270 100%);
        border-radius: 1rem;
        color: white;
        padding: 1.35rem 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 14px 32px rgba(7, 26, 47, 0.16);
    }

    .prediction-result small {
        color: #90E6F7;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    .prediction-result strong {
        display: block;
        font-size: 2rem;
        margin-top: .25rem;
    }

    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {
        border: 0;
        border-radius: 0.75rem;
        color: white;
        font-weight: 700;
        background: linear-gradient(90deg, #1473E6, #0AA8D8);
        min-height: 2.8rem;
    }

    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        color: white;
        box-shadow: 0 8px 20px rgba(20, 115, 230, 0.25);
    }

    @media (max-width: 760px) {
        .block-container { padding: 1rem 0.8rem 2rem; }
        .hero { padding: 1.5rem; }
        .hero h1 { font-size: 2rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def style_chart(figure, height=410):
    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#31475B"),
        title=dict(font=dict(size=18, color=NAVY), x=0.02),
        margin=dict(l=28, r=20, t=65, b=35),
        hoverlabel=dict(bgcolor=NAVY, font_color="white"),
        xaxis=dict(showgrid=False, linecolor="#DCE6EF"),
        yaxis=dict(gridcolor="#E8EFF6", zeroline=False),
        legend_title_text="",
    )
    return figure


def page_heading(title, description):
    st.markdown(
        f'<div class="section-title">{title}</div>'
        f'<div class="section-copy">{description}</div>',
        unsafe_allow_html=True,
    )


try:
    data = load_data()
except FileNotFoundError:
    st.error(f"The processed dataset could not be found at {DATA_PATH}.")
    st.stop()
except Exception as error:
    st.error(f"The processed dataset could not be loaded: {error}")
    st.stop()


st.sidebar.markdown("## 🛡️ CyberShield")
st.sidebar.caption("Incident intelligence workspace")
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Data Exploration",
        "Risk Profiles",
        "Resolution-Time Prediction",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Dataset coverage: 2015–2024")
st.sidebar.caption("Prediction API: Render")


st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Cyber Risk Intelligence</div>
        <h1>Cybersecurity Incident Analytics</h1>
        <p>Explore global attack patterns, compare data-driven risk profiles,
        and estimate incident resolution time through a deployed prediction API.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


if page == "Overview":
    page_heading(
        "Executive Overview",
        "A concise view of dataset coverage and the analytical outputs produced in Part I.",
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Total incidents", f"{len(data):,}")
    metric_2.metric("Countries", f"{data['country'].nunique():,}")
    metric_3.metric("Attack types", f"{data['attack_type'].nunique():,}")
    metric_4.metric("Risk profiles", f"{data['risk_profile'].nunique():,}")

    page_heading(
        "Incident Snapshot",
        "A sample of the processed and clustered records used by the application.",
    )
    overview_columns = [
        "country",
        "year",
        "attack_type",
        "target_industry",
        "financial_loss_million",
        "affected_users",
        "resolution_time_hours",
        "risk_profile",
    ]
    st.dataframe(
        data[overview_columns].head(12),
        use_container_width=True,
        hide_index=True,
        height=420,
        column_config={
            "country": "Country",
            "year": "Year",
            "attack_type": "Attack Type",
            "target_industry": "Industry",
            "financial_loss_million": st.column_config.NumberColumn(
                "Loss (USD M)", format="%.2f"
            ),
            "affected_users": st.column_config.NumberColumn(
                "Affected Users", format="%d"
            ),
            "resolution_time_hours": st.column_config.NumberColumn(
                "Resolution (Hours)", format="%.2f"
            ),
            "risk_profile": "Risk Profile",
        },
    )

    st.markdown(
        """
        <div class="model-note"><strong>Model performance note.</strong>
        The Random Forest achieved RMSE 20.77 hours, MAE 18.04 hours and
        R² −0.0124. It did not outperform the mean baseline, so predictions
        should be treated as exploratory estimates rather than operational guarantees.</div>
        """,
        unsafe_allow_html=True,
    )


elif page == "Data Exploration":
    page_heading(
        "Data Exploration",
        "Filter the dataset and compare incident volume and financial impact across attack types.",
    )

    selected_country = st.multiselect(
        "Filter by country",
        sorted(data["country"].dropna().unique()),
        placeholder="All countries",
    )
    filtered_data = data.copy()
    if selected_country:
        filtered_data = filtered_data[
            filtered_data["country"].isin(selected_country)
        ]

    filter_metric_1, filter_metric_2, filter_metric_3 = st.columns(3)
    filter_metric_1.metric("Matching incidents", f"{len(filtered_data):,}")
    filter_metric_2.metric(
        "Average loss",
        f"USD {filtered_data['financial_loss_million'].mean():,.2f}M",
    )
    filter_metric_3.metric(
        "Average resolution",
        f"{filtered_data['resolution_time_hours'].mean():,.1f} hours",
    )

    incidents_by_year = (
        filtered_data.groupby("year", as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
    )
    year_chart = px.area(
        incidents_by_year,
        x="year",
        y="incident_count",
        markers=True,
        title="Incident Volume by Year",
        labels={"year": "Year", "incident_count": "Incidents"},
        color_discrete_sequence=[BLUE],
    )
    year_chart.update_traces(line=dict(width=3), fillcolor="rgba(20,115,230,.14)")
    st.plotly_chart(style_chart(year_chart), use_container_width=True)

    chart_column_1, chart_column_2 = st.columns(2)
    attack_distribution = (
        filtered_data["attack_type"]
        .value_counts()
        .rename_axis("attack_type")
        .reset_index(name="incident_count")
    )
    attack_chart = px.bar(
        attack_distribution,
        x="incident_count",
        y="attack_type",
        orientation="h",
        title="Attack-Type Distribution",
        labels={"attack_type": "Attack Type", "incident_count": "Incidents"},
        color="incident_count",
        color_continuous_scale=["#BDEFFC", CYAN, BLUE],
    )
    attack_chart.update_layout(coloraxis_showscale=False)
    chart_column_1.plotly_chart(
        style_chart(attack_chart), use_container_width=True
    )

    average_loss = (
        filtered_data.groupby("attack_type", as_index=False)[
            "financial_loss_million"
        ]
        .mean()
        .sort_values("financial_loss_million", ascending=True)
    )
    loss_chart = px.bar(
        average_loss,
        x="financial_loss_million",
        y="attack_type",
        orientation="h",
        title="Average Financial Loss by Attack Type",
        labels={
            "attack_type": "Attack Type",
            "financial_loss_million": "Average Loss (USD M)",
        },
        color_discrete_sequence=[TEAL],
    )
    chart_column_2.plotly_chart(style_chart(loss_chart), use_container_width=True)

    with st.expander("View filtered incident records"):
        st.dataframe(
            filtered_data,
            use_container_width=True,
            hide_index=True,
            height=430,
        )


elif page == "Risk Profiles":
    page_heading(
        "Cyber-Risk Profiles",
        "Four K-Means segments summarise financial impact, user exposure and resolution complexity.",
    )

    profile_summary = (
        data.groupby("risk_profile", as_index=False)
        .agg(
            incident_count=("cluster_id", "size"),
            average_financial_loss_million=("financial_loss_million", "mean"),
            average_affected_users=("affected_users", "mean"),
            average_resolution_time_hours=("resolution_time_hours", "mean"),
        )
    )
    profile_summary["average_financial_loss_million"] = profile_summary[
        "average_financial_loss_million"
    ].round(2)
    profile_summary["average_affected_users"] = profile_summary[
        "average_affected_users"
    ].round(0)
    profile_summary["average_resolution_time_hours"] = profile_summary[
        "average_resolution_time_hours"
    ].round(2)

    profile_chart = px.bar(
        profile_summary.sort_values("incident_count", ascending=True),
        x="incident_count",
        y="risk_profile",
        orientation="h",
        color="risk_profile",
        title="Incident Count by Risk Profile",
        labels={"risk_profile": "Risk Profile", "incident_count": "Incidents"},
        color_discrete_sequence=PALETTE,
    )
    profile_chart.update_layout(showlegend=False)
    st.plotly_chart(style_chart(profile_chart, height=440), use_container_width=True)

    st.dataframe(
        profile_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "risk_profile": "Risk Profile",
            "incident_count": "Incidents",
            "average_financial_loss_million": st.column_config.NumberColumn(
                "Avg. Loss (USD M)", format="%.2f"
            ),
            "average_affected_users": st.column_config.NumberColumn(
                "Avg. Affected Users", format="%d"
            ),
            "average_resolution_time_hours": st.column_config.NumberColumn(
                "Avg. Resolution (Hours)", format="%.2f"
            ),
        },
    )

    page_heading("Profile Interpretation", "How each cluster should be read.")
    interpretation_1, interpretation_2 = st.columns(2)
    interpretation_1.info(
        "**High-Impact Rapid-Response Incidents**  \n"
        "Above-average loss and user impact, with shorter resolution time."
    )
    interpretation_1.info(
        "**Widespread Low-Loss Incidents**  \n"
        "Broad user impact but comparatively low financial loss."
    )
    interpretation_2.info(
        "**High-Impact Long-Resolution Incidents**  \n"
        "High financial and user impact with longer resolution requirements."
    )
    interpretation_2.info(
        "**Concentrated High-Cost Incidents**  \n"
        "Fewer affected users but high financial loss per affected user."
    )


elif page == "Resolution-Time Prediction":
    page_heading(
        "Resolution-Time Prediction",
        "Enter a new incident. The app sends the information to the public FastAPI endpoint on Render.",
    )

    with st.form("prediction_form"):
        input_column_1, input_column_2 = st.columns(2)

        with input_column_1:
            country = st.selectbox(
                "Country", sorted(data["country"].dropna().unique())
            )
            year = st.number_input(
                "Year", min_value=2015, max_value=2030, value=2024, step=1
            )
            attack_type = st.selectbox(
                "Attack type", sorted(data["attack_type"].dropna().unique())
            )
            target_industry = st.selectbox(
                "Target industry",
                sorted(data["target_industry"].dropna().unique()),
            )
            financial_loss = st.number_input(
                "Financial loss (Million USD)",
                min_value=0.0,
                value=50.0,
                step=1.0,
            )

        with input_column_2:
            affected_users = st.number_input(
                "Number of affected users",
                min_value=1,
                value=100000,
                step=1000,
            )
            attack_source = st.selectbox(
                "Attack source", sorted(data["attack_source"].dropna().unique())
            )
            vulnerability_type = st.selectbox(
                "Vulnerability type",
                sorted(data["vulnerability_type"].dropna().unique()),
            )
            defense_mechanism = st.selectbox(
                "Defense mechanism",
                sorted(data["defense_mechanism"].dropna().unique()),
            )

        submit_prediction = st.form_submit_button(
            "Predict resolution time", use_container_width=True
        )

    if submit_prediction:
        request_payload = {
            "country": country,
            "year": int(year),
            "attack_type": attack_type,
            "target_industry": target_industry,
            "financial_loss_million": float(financial_loss),
            "affected_users": int(affected_users),
            "attack_source": attack_source,
            "vulnerability_type": vulnerability_type,
            "defense_mechanism": defense_mechanism,
        }

        try:
            with st.spinner("Requesting prediction from Render..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json=request_payload,
                    timeout=60,
                )
            response.raise_for_status()
            result = response.json()
            predicted_hours = result["predicted_resolution_time_hours"]

            st.markdown(
                f"""
                <div class="prediction-result">
                    <small>Estimated resolution time</small>
                    <strong>{predicted_hours:.2f} hours</strong>
                    <span>Generated by the Part I Random Forest model served through FastAPI.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "Interpret this estimate cautiously: the model did not outperform "
                "the mean baseline during evaluation."
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "The prediction API is unavailable. The free Render service may be waking up; please try again shortly."
            )
        except requests.exceptions.Timeout:
            st.error(
                "The prediction request timed out. Please wait briefly and try again."
            )
        except requests.exceptions.HTTPError:
            st.error(f"The API rejected the request: {response.text}")
        except (KeyError, TypeError, ValueError) as error:
            st.error(f"The API returned an unexpected response: {error}")
