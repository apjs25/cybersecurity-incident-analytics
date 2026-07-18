import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Cybersecurity Resolution-Time API",
    description=(
        "Predicts incident resolution time using "
        "the Spark ML model trained in Part I."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# Model path
# --------------------------------------------------

BASE_DIRECTORY = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(
        BASE_DIRECTORY,
        "model",
        "cyber_resolution_pipeline",
    ),
)

print("Base directory:", BASE_DIRECTORY)
print("Model path:", MODEL_PATH)


if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(
        f"Spark model directory was not found: {MODEL_PATH}"
    )


# --------------------------------------------------
# Spark session and model
# --------------------------------------------------

spark = (
    SparkSession.builder
    .master("local[1]")
    .appName("CybersecurityPredictionAPI")
    .config("spark.driver.memory", "384m")
    .config("spark.executor.memory", "384m")
    .config("spark.driver.maxResultSize", "64m")
    .config("spark.sql.shuffle.partitions", "1")
    .config("spark.default.parallelism", "1")
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("Spark session is ready.")

model = PipelineModel.load(MODEL_PATH)

print("Model loaded successfully.")


# --------------------------------------------------
# Request schema
# --------------------------------------------------

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


# --------------------------------------------------
# API endpoints
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "online",
        "message": (
            "Cybersecurity resolution-time "
            "prediction API"
        ),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/predict")
def predict(incident: IncidentInput):
    try:
        record = incident.model_dump()

        record["loss_per_user"] = (
            record["financial_loss_million"]
            * 1_000_000
            / record["affected_users"]
        )

        input_df = spark.createDataFrame(
            [record]
        )

        prediction_df = model.transform(
            input_df
        )

        prediction_row = (
            prediction_df
            .select("prediction")
            .first()
        )

        if prediction_row is None:
            raise ValueError(
                "The model returned no prediction."
            )

        predicted_hours = max(
            0.0,
            float(prediction_row["prediction"]),
        )

        return {
            "predicted_resolution_time_hours": round(
                predicted_hours,
                2,
            ),
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction could not be generated: "
                f"{str(error)}"
            ),
        )
