import json
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIRECTORY = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIRECTORY,
    "model",
    "cyber_resolution_model.json"
)


CATEGORICAL_COLUMNS = [
    "country",
    "attack_type",
    "target_industry",
    "attack_source",
    "vulnerability_type",
    "defense_mechanism"
]

NUMERIC_COLUMNS = [
    "year",
    "financial_loss_million",
    "affected_users",
    "loss_per_user"
]


if not os.path.isfile(MODEL_PATH):
    raise RuntimeError(
        f"Model file was not found: {MODEL_PATH}"
    )


with open(
    MODEL_PATH,
    "r",
    encoding="utf-8"
) as model_file:
    model = json.load(model_file)


app = FastAPI(
    title="Cybersecurity Resolution-Time API",
    description=(
        "Serves a lightweight parameter export of "
        "the Spark Random Forest model trained in Part I."
    ),
    version="2.0.0"
)


class IncidentInput(BaseModel):
    country: str = Field(min_length=1)
    year: int = Field(ge=2015, le=2030)
    attack_type: str = Field(min_length=1)
    target_industry: str = Field(min_length=1)
    financial_loss_million: float = Field(ge=0)
    affected_users: int = Field(ge=1)
    attack_source: str = Field(min_length=1)
    vulnerability_type: str = Field(min_length=1)
    defense_mechanism: str = Field(min_length=1)


class PredictionOutput(BaseModel):
    predicted_resolution_time_hours: float


def build_features(record):
    features = model[
        "baseline_features"
    ].copy()

    for column_name in CATEGORICAL_COLUMNS:
        label = record[column_name]

        delta = (
            model["categorical_deltas"]
            [column_name]
            .get(label)
        )

        if delta is not None:
            features = [
                value + addition
                for value, addition
                in zip(features, delta)
            ]

    for column_name in NUMERIC_COLUMNS:
        amount = float(record[column_name])

        delta = model[
            "numeric_deltas"
        ][column_name]

        features = [
            value + amount * addition
            for value, addition
            in zip(features, delta)
        ]

    return features


def predict_tree(node, features):
    if node["type"] == "leaf":
        return node["prediction"]

    feature_value = features[
        node["feature_index"]
    ]

    if node["split_type"] == "continuous":
        go_left = (
            feature_value <= node["threshold"]
        )
    else:
        go_left = (
            feature_value
            in node["left_categories"]
        )

    if go_left:
        return predict_tree(
            node["left"],
            features
        )

    return predict_tree(
        node["right"],
        features
    )


def predict_forest(features):
    weighted_total = 0.0
    weight_total = 0.0

    for tree, weight in zip(
        model["trees"],
        model["tree_weights"]
    ):
        weighted_total += (
            predict_tree(tree, features)
            * weight
        )

        weight_total += weight

    return weighted_total / weight_total


@app.get("/")
def root():
    return {
        "status": "online",
        "message": (
            "Cybersecurity resolution-time "
            "prediction API"
        )
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_source": (
            "Part I Spark Random Forest"
        )
    }


@app.post(
    "/predict",
    response_model=PredictionOutput
)
def predict(incident: IncidentInput):
    try:
        record = incident.model_dump()

        record["loss_per_user"] = (
            record["financial_loss_million"]
            * 1_000_000
            / record["affected_users"]
        )

        features = build_features(record)

        predicted_hours = predict_forest(
            features
        )

        predicted_hours = max(
            0.0,
            float(predicted_hours)
        )

        return {
            "predicted_resolution_time_hours": round(
                predicted_hours,
                2
            )
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction could not be generated: "
                f"{str(error)}"
            )
        )
       