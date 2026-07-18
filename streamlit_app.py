import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Cybersecurity Incident Analytics",
    page_icon="🛡️",
    layout="wide",
)


# --------------------------------------------------
# Paths and API configuration
# --------------------------------------------------

BASE_DIRECTORY = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIRECTORY,
    "data",
    "clustered_incidents.csv",
)

API_URL = os.getenv(
    "API_URL",
    "https://cybersecurity-resolution-api.onrender.com"
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    data = load_data()

except FileNotFoundError:
    st.error(
        "The processed dataset could not be found at "
        f"{DATA_PATH}."
    )
    st.stop()

except Exception as error:
    st.error(
        "The processed dataset could not be loaded: "
        f"{error}"
    )
    st.stop()


# --------------------------------------------------
# Header and navigation
# --------------------------------------------------

st.title("Cybersecurity Incident Analytics")
st.caption(
    "Cyber-risk profiles and resolution-time prediction "
    "based on global cybersecurity incidents from 2015–2024."
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Data Exploration",
        "Risk Profiles",
        "Resolution-Time Prediction",
    ],
)


# --------------------------------------------------
# Overview page
# --------------------------------------------------

if page == "Overview":
    st.header("Overview")

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    metric_1.metric(
        "Total incidents",
        f"{len(data):,}",
    )

    metric_2.metric(
        "Countries",
        data["country"].nunique(),
    )

    metric_3.metric(
        "Attack types",
        data["attack_type"].nunique(),
    )

    metric_4.metric(
        "Risk profiles",
        data["risk_profile"].nunique(),
    )

    st.subheader("Dataset preview")

    st.dataframe(
        data.head(20),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "The Random Forest model achieved an RMSE of "
        "20.77 hours, an MAE of 18.04 hours, and an "
        "R² value of -0.0124. The model did not "
        "outperform the mean baseline, indicating that "
        "the available incident characteristics provide "
        "limited predictive information for resolution time."
    )


# --------------------------------------------------
# Data Exploration page
# --------------------------------------------------

elif page == "Data Exploration":
    st.header("Data Exploration")

    selected_country = st.multiselect(
        "Filter by country",
        sorted(data["country"].dropna().unique()),
    )

    filtered_data = data.copy()

    if selected_country:
        filtered_data = filtered_data[
            filtered_data["country"].isin(
                selected_country
            )
        ]

    incidents_by_year = (
        filtered_data.groupby(
            "year",
            as_index=False,
        )
        .size()
        .rename(
            columns={"size": "incident_count"}
        )
    )

    year_chart = px.line(
        incidents_by_year,
        x="year",
        y="incident_count",
        markers=True,
        title="Number of Incidents by Year",
        labels={
            "year": "Year",
            "incident_count": "Number of Incidents",
        },
    )

    st.plotly_chart(
        year_chart,
        use_container_width=True,
    )

    chart_column_1, chart_column_2 = st.columns(2)

    attack_distribution = (
        filtered_data["attack_type"]
        .value_counts()
        .rename_axis("attack_type")
        .reset_index(name="incident_count")
    )

    attack_chart = px.bar(
        attack_distribution,
        x="attack_type",
        y="incident_count",
        title="Incident Distribution by Attack Type",
        labels={
            "attack_type": "Attack Type",
            "incident_count": "Number of Incidents",
        },
    )

    chart_column_1.plotly_chart(
        attack_chart,
        use_container_width=True,
    )

    average_loss = (
        filtered_data.groupby(
            "attack_type",
            as_index=False,
        )["financial_loss_million"]
        .mean()
        .sort_values(
            "financial_loss_million",
            ascending=False,
        )
    )

    loss_chart = px.bar(
        average_loss,
        x="attack_type",
        y="financial_loss_million",
        title="Average Financial Loss by Attack Type",
        labels={
            "attack_type": "Attack Type",
            "financial_loss_million": (
                "Average Financial Loss (Million USD)"
            ),
        },
    )

    chart_column_2.plotly_chart(
        loss_chart,
        use_container_width=True,
    )

    st.subheader("Filtered data")

    st.dataframe(
        filtered_data,
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# Risk Profiles page
# --------------------------------------------------

elif page == "Risk Profiles":
    st.header("Cyber-Risk Profiles")

    profile_summary = (
        data.groupby(
            "risk_profile",
            as_index=False,
        )
        .agg(
            incident_count=(
                "cluster_id",
                "size",
            ),
            average_financial_loss_million=(
                "financial_loss_million",
                "mean",
            ),
            average_affected_users=(
                "affected_users",
                "mean",
            ),
            average_resolution_time_hours=(
                "resolution_time_hours",
                "mean",
            ),
        )
    )

    profile_summary[
        "average_financial_loss_million"
    ] = profile_summary[
        "average_financial_loss_million"
    ].round(2)

    profile_summary[
        "average_affected_users"
    ] = profile_summary[
        "average_affected_users"
    ].round(0)

    profile_summary[
        "average_resolution_time_hours"
    ] = profile_summary[
        "average_resolution_time_hours"
    ].round(2)

    profile_chart = px.bar(
        profile_summary,
        x="risk_profile",
        y="incident_count",
        color="risk_profile",
        title="Incident Count by Risk Profile",
        labels={
            "risk_profile": "Risk Profile",
            "incident_count": "Number of Incidents",
        },
    )

    profile_chart.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        profile_chart,
        use_container_width=True,
    )

    st.dataframe(
        profile_summary,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        **Risk-profile interpretation**

        - **High-Impact Rapid-Response Incidents:** Above-average
          financial loss and user impact, with shorter resolution time.
        - **Widespread Low-Loss Incidents:** Broad user impact but
          comparatively low financial loss.
        - **High-Impact Long-Resolution Incidents:** High financial and
          user impact combined with longer resolution requirements.
        - **Concentrated High-Cost Incidents:** Relatively few affected
          users but high financial loss per affected user.
        """
    )


# --------------------------------------------------
# Prediction page
# --------------------------------------------------

elif page == "Resolution-Time Prediction":
    st.header("Resolution-Time Prediction")

    st.write(
        "Enter the incident characteristics below. "
        "The application will send the information to "
        "the FastAPI prediction service."
    )

    with st.form("prediction_form"):
        input_column_1, input_column_2 = st.columns(2)

        with input_column_1:
            country = st.selectbox(
                "Country",
                sorted(data["country"].dropna().unique()),
            )

            year = st.number_input(
                "Year",
                min_value=2015,
                max_value=2030,
                value=2024,
                step=1,
            )

            attack_type = st.selectbox(
                "Attack type",
                sorted(
                    data["attack_type"]
                    .dropna()
                    .unique()
                ),
            )

            target_industry = st.selectbox(
                "Target industry",
                sorted(
                    data["target_industry"]
                    .dropna()
                    .unique()
                ),
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
                "Attack source",
                sorted(
                    data["attack_source"]
                    .dropna()
                    .unique()
                ),
            )

            vulnerability_type = st.selectbox(
                "Vulnerability type",
                sorted(
                    data["vulnerability_type"]
                    .dropna()
                    .unique()
                ),
            )

            defense_mechanism = st.selectbox(
                "Defense mechanism",
                sorted(
                    data["defense_mechanism"]
                    .dropna()
                    .unique()
                ),
            )

        submit_prediction = st.form_submit_button(
            "Predict resolution time",
            use_container_width=True,
        )

    if submit_prediction:
        request_payload = {
            "country": country,
            "year": int(year),
            "attack_type": attack_type,
            "target_industry": target_industry,
            "financial_loss_million": float(
                financial_loss
            ),
            "affected_users": int(affected_users),
            "attack_source": attack_source,
            "vulnerability_type": vulnerability_type,
            "defense_mechanism": defense_mechanism,
        }

        try:
            with st.spinner(
                "Requesting prediction..."
            ):
                response = requests.post(
                    f"{API_URL}/predict",
                    json=request_payload,
                    timeout=60,
                )

            response.raise_for_status()
            result = response.json()

            predicted_hours = result[
                "predicted_resolution_time_hours"
            ]

            st.success(
                "Predicted resolution time: "
                f"{predicted_hours:.2f} hours"
            )

            st.caption(
                "This estimate should be interpreted with "
                "caution because the model did not outperform "
                "the mean baseline during evaluation."
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "The prediction API is unavailable. "
                "Please confirm that FastAPI is running."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The prediction request timed out."
            )

        except requests.exceptions.HTTPError:
            st.error(
                "The API rejected the request: "
                f"{response.text}"
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            st.error(
                "The API returned an unexpected response: "
                f"{error}"
            )