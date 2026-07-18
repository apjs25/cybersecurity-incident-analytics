# Cybersecurity Incident Analytics

## Project Overview

This project analyses global cybersecurity incidents from 2015 to 2024. Apache Spark was used to clean and preprocess the data, segment incidents into cyber-risk profiles, and train a Random Forest regression model to predict incident resolution time.

Part II provides a FastAPI prediction service and a Streamlit dashboard for data exploration, risk-profile visualisation, and resolution-time prediction.

## Technologies

- Apache Spark and Spark MLlib
- Python
- FastAPI
- Streamlit
- Pandas
- Plotly
- Uvicorn

## Project Structure

```text
cybersecurity_project/
├── api/
│   ├── __init__.py
│   ├── app.py
│   └── model/
│       └── cyber_resolution_pipeline/
├── data/
│   └── clustered_incidents.csv
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Risk Profiles

K-Means clustering was evaluated using silhouette scores. Four clusters were selected and interpreted as:

- High-Impact Rapid-Response Incidents
- Widespread Low-Loss Incidents
- High-Impact Long-Resolution Incidents
- Concentrated High-Cost Incidents

## Prediction Model

A Spark Random Forest regression pipeline was trained to predict incident resolution time. The complete pipeline includes categorical indexing, one-hot encoding, feature assembly, and regression.

Model evaluation results:

- RMSE: 20.7732 hours
- MAE: 18.0449 hours
- R²: -0.0124

The model did not outperform the mean baseline. This indicates that the available incident characteristics contain limited predictive information about resolution time. Predictions should therefore be interpreted cautiously.

## Installation

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run FastAPI

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Local endpoints:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
```

## Run Streamlit

Open another terminal and run:

```bash
source .venv/bin/activate
python -m streamlit run streamlit_app.py
```

## API Request Example

```json
{
  "country": "Australia",
  "year": 2024,
  "attack_type": "DDoS",
  "target_industry": "Banking",
  "financial_loss_million": 50.0,
  "affected_users": 100000,
  "attack_source": "Hacker Group",
  "vulnerability_type": "Social Engineering",
  "defense_mechanism": "AI-based Detection"
}
```

Example response:

```json
{
  "predicted_resolution_time_hours": 32.75
}
```

## Deployment URL

To be updated after successful deployment.

## Known Limitations

- The regression model showed limited predictive performance.
- Important operational variables may be absent from the dataset.
- The Spark runtime requires Java and considerably more memory than a lightweight Python model.
- Predictions should be treated as analytical estimates rather than operational guarantees.