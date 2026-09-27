import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


INPUT_SHAPE = (128, 128, 13)

# Development-level classes present in the current dataset.
# These are encoded as:
# 0 = DB
# 1 = TS
NUM_DEVELOPMENT_CLASSES = 2


def build_model():

    inputs = keras.Input(
        shape=INPUT_SHAPE,
        name="multisource_satellite"
    )

    # -----------------------------------------------------
    # CNN FEATURE EXTRACTION
    # -----------------------------------------------------

    x = layers.Conv2D(
        32,
        3,
        padding="same",
        activation="relu"
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(
        64,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(
        128,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(
        256,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    # Convert spatial feature maps into a compact feature vector.
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.30)(x)

    # -----------------------------------------------------
    # MULTI-TASK OUTPUTS
    # -----------------------------------------------------

    # Cyclone intensity regression
    intensity_output = layers.Dense(
        1,
        name="intensity"
    )(x)

    # Central minimum pressure regression
    pressure_output = layers.Dense(
        1,
        name="pressure"
    )(x)

    # Cyclone development classification
    development_output = layers.Dense(
        NUM_DEVELOPMENT_CLASSES,
        activation="softmax",
        name="development"
    )(x)

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = keras.Model(
        inputs=inputs,
        outputs={
            "intensity": intensity_output,
            "pressure": pressure_output,
            "development": development_output,
        },
        name="CycloneMultiSourceCNN"
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss={
            "intensity": "mse",
            "pressure": "mse",
            "development": "sparse_categorical_crossentropy",
        },

        metrics={
            "intensity": ["mae"],
            "pressure": ["mae"],
            "development": ["accuracy"],
        }
    )

    return model


if __name__ == "__main__":

    model = build_model()

    model.summary()