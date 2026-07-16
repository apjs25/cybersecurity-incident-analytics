# Global Cybersecurity Incident Analytics and Resolution-Time Prediction

## Project Overview

This project analyses global cybersecurity incidents recorded from 2015 to 2024. Apache Spark was used to clean and preprocess the data, engineer features, segment incidents into cyber-risk profiles, and train a supervised model to predict incident resolution time.

Part II provides:

- A FastAPI prediction service
- A Streamlit analytics application
- Interactive data exploration
- Cyber-risk profile visualisation
- Incident resolution-time prediction

## Dataset

The dataset contains 3,000 cybersecurity incident records and 10 original attributes, including:

- Country and year
- Attack type and source
- Target industry
- Financial loss
- Number of affected users
- Vulnerability type
- Defence mechanism
- Incident resolution time

Data-quality checks found no missing values, duplicate records, invalid years, negative financial losses, negative affected-user counts, or invalid resolution times.

## Technologies

- Apache Spark and Spark MLlib
- Python
- FastAPI
- Streamlit
- Pandas
- Plotly
- Pydantic
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
└── README.md
```

## Incident Segmentation

K-Means clustering was applied using standardised numerical features. Values of `k` from 2 to 8 were evaluated using the silhouette score.

The selected solution used four clusters and achieved a silhouette score of 0.5100.

The resulting risk profiles were:

1. High-Impact Rapid-Response Incidents
2. Widespread Low-Loss Incidents
3. High-Impact Long-Resolution Incidents
4. Concentrated High-Cost Incidents

Attack types were examined after clustering. Their distributions were relatively balanced, so they were treated as supporting information rather than the main basis for profile naming.

## Predictive Model

A Spark ML Pipeline was created with:

- String indexing
- One-hot encoding
- Feature assembly
- Random Forest regression

The model predicts incident resolution time using the following characteristics:

- Country
- Year
- Attack type
- Target industry
- Financial loss
- Number of affected users
- Attack source
- Vulnerability type
- Defence mechanism
- Financial loss per affected user

## Model Evaluation

### Random Forest

- RMSE: 20.7732 hours
- MAE: 18.0449 hours
- R²: -0.0124

### Mean Baseline

- RMSE: 20.6466 hours
- MAE: 17.8800 hours
- R²: -0.0001

The Random Forest did not outperform the mean baseline. This indicates that the available incident characteristics contain limited predictive information about resolution time.

The residual analysis showed that predictions were concentrated near the overall mean. Very short incidents were often over-predicted, while very long incidents were frequently under-predicted. Prediction errors were slightly larger for Phishing incidents than for Ransomware incidents. Very-high-loss incidents were slightly over-predicted on average rather than consistently under-predicted.

Possible missing predictors include incident severity, response-team capacity, containment delay, organisational readiness, and system complexity.

## Installation

Python and Java are required because the prediction API loads a Spark ML model.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Run FastAPI Locally

From the project root directory:

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

## API Request Example

Endpoint:

```text
POST /predict
```

Request body:

```json
{
  "country": "China",
  "year": 2024,
  "attack_type": "Ransomware",
  "target_industry": "Banking",
  "financial_loss_million": 50.0,
  "affected_users": 100000,
  "attack_source": "Hacker Group",
  "vulnerability_type": "Unpatched Software",
  "defense_mechanism": "Firewall"
}
```

Example response:

```json
{
  "predicted_resolution_time_hours": 35.78
}
```

## Run Streamlit Locally

Start FastAPI first. Open a second terminal, activate the same virtual environment, and run:

```bash
python -m streamlit run streamlit_app.py
```

Open:

```text
http://localhost:8501
```

The Streamlit application contains:

- Overview
- Data Exploration
- Risk Profiles
- Resolution-Time Prediction

## Deployment

### Render FastAPI URL

```text
TO BE ADDED AFTER DEPLOYMENT
```

### API Documentation URL

```text
TO BE ADDED AFTER DEPLOYMENT/docs
```

The Streamlit application reads the API address from the `API_URL` environment variable. When this variable is absent, it uses the local API at `http://127.0.0.1:8000`.

## Input Validation

The FastAPI service validates:

- Required categorical values
- Year range
- Non-negative financial loss
- Positive affected-user count

Invalid requests receive an appropriate validation response.

## Known Limitations

- The dataset appears to contain weak relationships between the available predictors and resolution time.
- The model does not outperform the mean baseline.
- Predictions should therefore be treated as experimental estimates rather than operational guarantees.
- Loading PySpark and the complete Spark model may require considerable memory and startup time on cloud platforms.
- Categories not observed during training may have limited predictive meaning even though the pipeline can process them.

## Local Testing

The API and Streamlit application were tested locally. A valid prediction request returned HTTP status 200, and Streamlit successfully displayed the result returned by FastAPI.