from pathlib import Path
import json

import numpy as np
from tensorflow import keras


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

X_PATH = DATA_DIR / "X_multisource.npy"
METADATA_PATH = DATA_DIR / "metadata.json"
CHANNELS_PATH = DATA_DIR / "channels.json"

MODEL_PATH = MODEL_DIR / "cyclone_multisource_cnn.keras"
LABELS_PATH = MODEL_DIR / "development_labels.json"


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def load_dataset():
    """Load the multi-source satellite tensor."""
    if not X_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {X_PATH}")

    return np.load(X_PATH)


def load_metadata():
    """Load observation metadata."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_channels():
    """Load channel definitions."""
    if not CHANNELS_PATH.exists():
        return {}

    with open(CHANNELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_labels():
    """Load development-class labels."""
    if not LABELS_PATH.exists():
        return {}

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

def load_model():
    """Load the saved multi-task CNN."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return keras.models.load_model(MODEL_PATH)


# ---------------------------------------------------------
# OBSERVATIONS
# ---------------------------------------------------------

def observation_count(X):
    return int(len(X))


def get_observation(X, index):
    """Return one observation."""
    if index < 0 or index >= len(X):
        raise IndexError("Observation index out of range.")

    return X[index]


def get_metadata_item(metadata, index):
    """Return metadata for one observation."""
    if isinstance(metadata, list):
        return metadata[index]

    if isinstance(metadata, dict):
        observations = metadata.get("observations")

        if observations is not None:
            return observations[index]

    return {}


# ---------------------------------------------------------
# MODEL OUTPUTS
# ---------------------------------------------------------

def predict_observation(model, X, index):
    """Run the model on one observation."""
    sample = np.asarray(X[index:index + 1], dtype=np.float32)

    outputs = model.predict(sample, verbose=0)

    return extract_model_outputs(outputs)


def predict_all(model, X):
    """Run the model on all observations in one batch."""
    batch = np.asarray(X, dtype=np.float32)

    outputs = model.predict(batch, verbose=0)

    return extract_batch_outputs(outputs)


def extract_model_outputs(outputs):
    """Normalize Keras multi-output predictions."""

    if isinstance(outputs, dict):
        intensity = outputs.get("intensity")
        pressure = outputs.get("pressure")
        development = outputs.get("development")
    else:
        intensity = outputs[0]
        pressure = outputs[1]
        development = outputs[2]

    return {
        "intensity": float(np.asarray(intensity).reshape(-1)[0]),
        "pressure": float(np.asarray(pressure).reshape(-1)[0]),
        "development": np.asarray(development)[0].tolist(),
    }


def extract_batch_outputs(outputs):
    """Convert batch predictions into a list of dictionaries."""

    if isinstance(outputs, dict):
        intensity = np.asarray(outputs["intensity"]).reshape(-1)
        pressure = np.asarray(outputs["pressure"]).reshape(-1)
        development = np.asarray(outputs["development"])
    else:
        intensity = np.asarray(outputs[0]).reshape(-1)
        pressure = np.asarray(outputs[1]).reshape(-1)
        development = np.asarray(outputs[2])

    results = []

    for i in range(len(intensity)):
        results.append(
            {
                "index": i,
                "intensity": float(intensity[i]),
                "pressure": float(pressure[i]),
                "development": development[i].tolist(),
            }
        )

    return results


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def get_value(item, *keys, default=None):
    """Safely retrieve a value from metadata."""

    if not isinstance(item, dict):
        return default

    for key in keys:
        if key in item:
            value = item[key]

            if value is not None:
                return value

    return default


def format_number(value, digits=2):
    """Format numeric values safely."""

    if value is None:
        return "N/A"

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def development_label(probabilities, labels=None):
    """Return the predicted development label."""

    if probabilities is None or len(probabilities) == 0:
        return "Unknown"

    index = int(np.argmax(probabilities))

    if isinstance(labels, dict):
        # Handle either index -> label or label -> index formats.
        if str(index) in labels:
            return str(labels[str(index)])

        for label, value in labels.items():
            try:
                if int(value) == index:
                    return str(label)
            except (TypeError, ValueError):
                pass

    return f"Class {index}"


# ---------------------------------------------------------
# DATASET SUMMARY
# ---------------------------------------------------------

def dataset_summary(X, metadata):
    """Return basic dataset statistics."""

    finite_ratio = float(np.isfinite(X).mean()) if X.size else 0.0

    return {
        "observations": int(len(X)),
        "height": int(X.shape[1]) if X.ndim >= 2 else 0,
        "width": int(X.shape[2]) if X.ndim >= 3 else 0,
        "channels": int(X.shape[3]) if X.ndim >= 4 else 0,
        "finite_percent": finite_ratio * 100,
        "metadata_records": len(metadata) if hasattr(metadata, "__len__") else 0,
    }


# ---------------------------------------------------------
# SINGLE + BATCH ANALYSIS
# ---------------------------------------------------------

def analyze_single(model, X, metadata, labels, index):
    """Complete analysis for one observation."""

    prediction = predict_observation(model, X, index)
    meta = get_metadata_item(metadata, index)

    prediction["index"] = index
    prediction["label"] = development_label(
        prediction["development"],
        labels,
    )
    prediction["metadata"] = meta

    return prediction


def analyze_all(model, X, metadata, labels):
    """Complete analysis for every observation."""

    predictions = predict_all(model, X)

    results = []

    for prediction in predictions:
        index = prediction["index"]

        prediction["label"] = development_label(
            prediction["development"],
            labels,
        )

        prediction["metadata"] = get_metadata_item(
            metadata,
            index,
        )

        results.append(prediction)

    return results