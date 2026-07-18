import json
import os

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SPARK_MODEL_PATH = os.path.join(
    BASE_DIR,
    "api",
    "model",
    "cyber_resolution_pipeline"
)

JSON_MODEL_PATH = os.path.join(
    BASE_DIR,
    "api",
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


schema = StructType([
    StructField("country", StringType(), False),
    StructField("year", DoubleType(), False),
    StructField("attack_type", StringType(), False),
    StructField("target_industry", StringType(), False),
    StructField(
        "financial_loss_million",
        DoubleType(),
        False
    ),
    StructField("affected_users", DoubleType(), False),
    StructField("attack_source", StringType(), False),
    StructField(
        "vulnerability_type",
        StringType(),
        False
    ),
    StructField(
        "defense_mechanism",
        StringType(),
        False
    ),
    StructField("loss_per_user", DoubleType(), False)
])


spark = (
    SparkSession.builder
    .master("local[1]")
    .appName("ExportSparkModel")
    .config("spark.ui.enabled", "false")
    .config("spark.sql.shuffle.partitions", "1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

pipeline_model = PipelineModel.load(
    SPARK_MODEL_PATH
)

rf_model = pipeline_model.stages[-1]


base_record = {
    "country": "__UNKNOWN__",
    "year": 0.0,
    "attack_type": "__UNKNOWN__",
    "target_industry": "__UNKNOWN__",
    "financial_loss_million": 0.0,
    "affected_users": 0.0,
    "attack_source": "__UNKNOWN__",
    "vulnerability_type": "__UNKNOWN__",
    "defense_mechanism": "__UNKNOWN__",
    "loss_per_user": 0.0
}


def get_features(record):
    input_df = spark.createDataFrame(
        [record],
        schema=schema
    )

    features = (
        pipeline_model
        .transform(input_df)
        .select("features")
        .first()["features"]
    )

    return [
        float(value)
        for value in features.toArray()
    ]


def subtract_vectors(first, second):
    return [
        float(a - b)
        for a, b in zip(first, second)
    ]


baseline_features = get_features(
    base_record.copy()
)

feature_size = len(baseline_features)


indexer_labels = {}

for stage in pipeline_model.stages:
    if stage.__class__.__name__ == "StringIndexerModel":
        indexer_labels[stage.getInputCol()] = list(
            stage.labels
        )


categorical_deltas = {}

for column_name in CATEGORICAL_COLUMNS:
    categorical_deltas[column_name] = {}

    for label in indexer_labels[column_name]:
        test_record = base_record.copy()
        test_record[column_name] = label

        label_features = get_features(
            test_record
        )

        categorical_deltas[column_name][label] = (
            subtract_vectors(
                label_features,
                baseline_features
            )
        )


numeric_deltas = {}

for column_name in NUMERIC_COLUMNS:
    test_record = base_record.copy()
    test_record[column_name] = 1.0

    numeric_features = get_features(
        test_record
    )

    numeric_deltas[column_name] = subtract_vectors(
        numeric_features,
        baseline_features
    )


def export_tree_node(java_node):
    node_type = (
        java_node
        .getClass()
        .getSimpleName()
    )

    if node_type == "LeafNode":
        return {
            "type": "leaf",
            "prediction": float(
                java_node.prediction()
            )
        }

    split = java_node.split()

    split_type = (
        split
        .getClass()
        .getSimpleName()
    )

    exported_node = {
        "type": "internal",
        "feature_index": int(
            split.featureIndex()
        ),
        "left": export_tree_node(
            java_node.leftChild()
        ),
        "right": export_tree_node(
            java_node.rightChild()
        )
    }

    if split_type == "ContinuousSplit":
        exported_node["split_type"] = "continuous"
        exported_node["threshold"] = float(
            split.threshold()
        )

    elif split_type == "CategoricalSplit":
        exported_node["split_type"] = "categorical"
        exported_node["left_categories"] = [
            float(value)
            for value in split.leftCategories()
        ]

    else:
        raise RuntimeError(
            f"Unsupported split type: {split_type}"
        )

    return exported_node


trees = []

for tree in rf_model.trees:
    trees.append(
        export_tree_node(
            tree._java_obj.rootNode()
        )
    )


exported_model = {
    "model_type": "Spark RandomForestRegressor",
    "source": "Part I Spark PipelineModel",
    "feature_size": feature_size,
    "baseline_features": baseline_features,
    "categorical_deltas": categorical_deltas,
    "numeric_deltas": numeric_deltas,
    "tree_weights": [
        float(weight)
        for weight in rf_model.treeWeights
    ],
    "trees": trees
}


with open(
    JSON_MODEL_PATH,
    "w",
    encoding="utf-8"
) as model_file:
    json.dump(
        exported_model,
        model_file,
        separators=(",", ":")
    )


def build_lightweight_features(record):
    features = exported_model[
        "baseline_features"
    ].copy()

    for column_name in CATEGORICAL_COLUMNS:
        label = record[column_name]

        delta = (
            exported_model["categorical_deltas"]
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
        delta = exported_model[
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


def lightweight_predict(record):
    features = build_lightweight_features(
        record
    )

    weighted_total = 0.0
    weight_total = 0.0

    for tree, weight in zip(
        exported_model["trees"],
        exported_model["tree_weights"]
    ):
        weighted_total += (
            predict_tree(tree, features)
            * weight
        )
        weight_total += weight

    return weighted_total / weight_total


validation_records = [
    {
        "country": "Australia",
        "year": 2024.0,
        "attack_type": "DDoS",
        "target_industry": "Banking",
        "financial_loss_million": 50.0,
        "affected_users": 100000.0,
        "attack_source": "Hacker Group",
        "vulnerability_type": "Social Engineering",
        "defense_mechanism": "AI-based Detection"
    },
    {
        "country": "Malaysia",
        "year": 2020.0,
        "attack_type": "Ransomware",
        "target_industry": "Healthcare",
        "financial_loss_million": 80.0,
        "affected_users": 500000.0,
        "attack_source": "Insider",
        "vulnerability_type": "Weak Passwords",
        "defense_mechanism": "Firewall"
    }
]


for record in validation_records:
    record["loss_per_user"] = (
        record["financial_loss_million"]
        * 1_000_000
        / record["affected_users"]
    )

    spark_prediction = (
        pipeline_model
        .transform(
            spark.createDataFrame(
                [record],
                schema=schema
            )
        )
        .select("prediction")
        .first()["prediction"]
    )

    lightweight_prediction = lightweight_predict(
        record
    )

    difference = abs(
        float(spark_prediction)
        - lightweight_prediction
    )

    print("Spark prediction:", spark_prediction)
    print(
        "Lightweight prediction:",
        lightweight_prediction
    )
    print("Difference:", difference)

    if difference > 0.000001:
        raise RuntimeError(
            "Validation failed: predictions differ."
        )


print("Model exported successfully:")
print(JSON_MODEL_PATH)
print("All validation predictions matched.")

spark.stop()