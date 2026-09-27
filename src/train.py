import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"
MODEL_DIR = "models"

X_FILE = os.path.join(DATA_DIR, "X_multisource.npy")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "cyclone_multitask.keras"
)

LABEL_FILE = os.path.join(
    MODEL_DIR,
    "development_labels.json"
)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("CYCLONE AI TRAINING PIPELINE")
print("=" * 70)

X = np.load(X_FILE)

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:
    metadata = json.load(file)

print("Input shape:", X.shape)
print("Samples:", len(X))


# ============================================================
# PREPARE TARGETS
# ============================================================

development_labels = [
    item["development_level"]
    for item in metadata
]

intensity = np.array(
    [
        item["intensity"]
        for item in metadata
    ],
    dtype=np.float32
)

pressure = np.array(
    [
        item["central_min_pressure"]
        for item in metadata
    ],
    dtype=np.float32
)


# ============================================================
# ENCODE DEVELOPMENT LEVEL
# ============================================================

encoder = LabelEncoder()

y_class = encoder.fit_transform(
    development_labels
)

class_names = encoder.classes_.tolist()

print()
print("Development classes:", class_names)

for name in class_names:
    print(
        f"  {name}: "
        f"{development_labels.count(name)} samples"
    )


# ============================================================
# IMPORTANT SMALL-DATA CHECK
# ============================================================

if len(X) < 20:

    print()
    print("=" * 70)
    print("SMALL DATASET WARNING")
    print("=" * 70)
    print(
        "Only",
        len(X),
        "samples are currently available."
    )
    print(
        "A meaningful train/test accuracy cannot be established."
    )
    print(
        "The model will only be initialized and checked."
    )
    print("=" * 70)

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    from model import build_model

    model = build_model()

    model.save(
        MODEL_FILE
    )

    with open(
        LABEL_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            {
                "classes": class_names
            },
            file,
            indent=2
        )

    print()
    print("Model initialized and saved:")
    print(MODEL_FILE)

    print()
    print("Classification labels saved:")
    print(LABEL_FILE)

    print()
    print("Training stopped safely because the dataset is too small.")

    raise SystemExit(0)


# ============================================================
# NORMAL TRAINING
# ============================================================

from model import build_model

model = build_model()

model.summary()

history = model.fit(
    X,
    {
        "intensity": intensity,
        "pressure": pressure,
        "development": y_class
    },
    epochs=20,
    batch_size=4,
    validation_split=0.2,
    verbose=1
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

model.save(
    MODEL_FILE
)

with open(
    LABEL_FILE,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        {
            "classes": class_names
        },
        file,
        indent=2
    )

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print("Model:", MODEL_FILE)
print("Labels:", LABEL_FILE)