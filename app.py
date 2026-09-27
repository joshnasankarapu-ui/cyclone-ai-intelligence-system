import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow import keras


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

X_PATH = DATA_DIR / "X_multisource.npy"
METADATA_PATH = DATA_DIR / "metadata.json"
CHANNELS_PATH = DATA_DIR / "channels.json"

MODEL_PATH = MODEL_DIR / "cyclone_multitask.keras"
LABELS_PATH = MODEL_DIR / "development_labels.json"

st.set_page_config(
    page_title="Cyclone AI Intelligence",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM UI STYLE
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1rem;
    }

    .section-label {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
    }

    .prototype-box {
        padding: 0.9rem 1rem;
        border-radius: 10px;
        border: 1px solid #94a3b8;
        background-color: rgba(148, 163, 184, 0.08);
        margin: 0.6rem 0 1rem 0;
    }

    .info-box {
        padding: 0.9rem 1rem;
        border-radius: 10px;
        border: 1px solid #94a3b8;
        background-color: rgba(148, 163, 184, 0.08);
        margin: 0.6rem 0 1rem 0;
    }

    .pipeline-box {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #cbd5e1;
        min-height: 150px;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_dataset():

    X = np.load(X_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(CHANNELS_PATH, "r", encoding="utf-8") as f:
        channels = json.load(f)

    return X, metadata, channels


@st.cache_resource
def load_model():

    if MODEL_PATH.exists():
        return keras.models.load_model(MODEL_PATH)

    return None


@st.cache_data
def load_labels():

    if LABELS_PATH.exists():

        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            raw_labels = json.load(f)

        return {
            str(key): str(value)
            for key, value in raw_labels.items()
        }

    return {
        "0": "DB",
        "1": "TS",
    }


# =========================================================
# HELPERS
# =========================================================

def get_value(record, key, default="N/A"):

    value = record.get(key, default)

    if value is None:
        return default

    try:

        if np.isnan(value):
            return default

    except Exception:
        pass

    return value


def format_number(value, decimals=2):

    try:
        return f"{float(value):.{decimals}f}"

    except Exception:
        return str(value)


def extract_model_outputs(prediction):
    """
    Supports Keras predictions returned as either:

    - dictionary
    - list / tuple

    Model output order:

    intensity
    pressure
    development
    """

    if isinstance(prediction, dict):

        development_output = prediction.get(
            "development"
        )

        intensity_output = prediction.get(
            "intensity"
        )

        pressure_output = prediction.get(
            "pressure"
        )

    elif isinstance(prediction, (list, tuple)):

        intensity_output = prediction[0]
        pressure_output = prediction[1]
        development_output = prediction[2]

    else:

        raise ValueError(
            "Unexpected model prediction format."
        )

    if development_output is None:

        raise ValueError(
            "Development output was not found."
        )

    probabilities = np.asarray(
        development_output
    )[0]

    intensity_value = None
    pressure_value = None

    if intensity_output is not None:

        intensity_value = float(
            np.asarray(
                intensity_output
            ).reshape(-1)[0]
        )

    if pressure_output is not None:

        pressure_value = float(
            np.asarray(
                pressure_output
            ).reshape(-1)[0]
        )

    return (
        probabilities,
        intensity_value,
        pressure_value,
    )


# =========================================================
# LOAD PROJECT
# =========================================================

try:

    X, metadata, channels = load_dataset()

    model = load_model()

    labels = load_labels()

except Exception as e:

    st.error(
        f"Unable to load project data: {e}"
    )

    st.stop()


if not labels or not all(
    str(i) in labels
    for i in range(2)
):

    labels = {
        "0": "DB",
        "1": "TS",
    }


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🌀 Cyclone AI Intelligence System'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI/ML-based multi-source satellite intelligence "
    "for tropical cyclone identification, classification "
    "and analysis"
    "</div>",
    unsafe_allow_html=True,
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "⚙️ System Controls"
)

st.sidebar.markdown(
    "### Data Source"
)

st.sidebar.info(
    "NOAA TC PRIMED\n\n"
    "Tropical Cyclone PRecipitation, Infrared, "
    "Microwave, and Environmental Dataset"
)

st.sidebar.markdown(
    "### Satellite Observation"
)

sample_count = len(X)

selected_index = st.sidebar.selectbox(
    "Select observation",
    range(sample_count),
    format_func=lambda i:
        f"Observation {i + 1}",
)

record = metadata[selected_index]

st.sidebar.markdown(
    "### Observation Metadata"
)

st.sidebar.write(
    f"**Instrument:** "
    f"{get_value(record, 'instrument_name')}"
)

st.sidebar.write(
    f"**Platform:** "
    f"{get_value(record, 'platform_name')}"
)

st.sidebar.write(
    f"**File:** "
    f"{get_value(record, 'filename')}"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Prototype system for "
    "Smart India Hackathon 2026"
)


# =========================================================
# OBSERVED CYCLONE INFORMATION
# =========================================================

development = get_value(
    record,
    "development_level"
)

intensity = get_value(
    record,
    "intensity"
)

pressure = get_value(
    record,
    "central_min_pressure"
)

latitude = get_value(
    record,
    "storm_latitude"
)

longitude = get_value(
    record,
    "storm_longitude"
)

st.subheader(
    "🌪️ Cyclone Observation Overview"
)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:

    st.metric(
        "Observed Development",
        development,
        border=True,
    )

with c2:

    st.metric(
        "Observed Intensity",
        format_number(
            intensity,
            1
        ),
        border=True,
    )

with c3:

    st.metric(
        "Central Pressure",
        f"{format_number(pressure, 1)} hPa",
        border=True,
    )

with c4:

    st.metric(
        "Latitude",
        format_number(
            latitude,
            3
        ),
        border=True,
    )

with c5:

    st.metric(
        "Longitude",
        format_number(
            longitude,
            3
        ),
        border=True,
    )


# =========================================================
# AI MODEL PREDICTION
# =========================================================

st.divider()

st.subheader(
    "🤖 AI Cyclone Analysis"
)

predicted_class = "Unavailable"

confidence = 0.0

development_probabilities = None

ai_intensity = None

ai_pressure = None


if model is not None:

    try:

        model_input = X[
            selected_index:
            selected_index + 1
        ]

        prediction = model.predict(
            model_input,
            verbose=0
        )

        (
            development_probabilities,
            ai_intensity,
            ai_pressure,
        ) = extract_model_outputs(
            prediction
        )

        predicted_index = int(
            np.argmax(
                development_probabilities
            )
        )

        predicted_class = labels.get(
            str(predicted_index),
            "Unknown"
        )

        confidence = (
            float(
                development_probabilities[
                    predicted_index
                ]
            )
            * 100
        )

    except Exception as e:

        st.error(
            "AI prediction could not be generated: "
            f"{e}"
        )

else:

    st.warning(
        "AI model is not available.",
        icon="⚠️",
    )


# =========================================================
# AI OUTPUT CARDS
# =========================================================

a1, a2, a3, a4 = st.columns(4)

with a1:

    st.metric(
        "AI Classification",
        predicted_class,
        border=True,
    )

with a2:

    st.metric(
        "Classification Confidence",
        f"{confidence:.1f}%",
        border=True,
    )

with a3:

    if ai_intensity is not None:

        st.metric(
            "AI Intensity",
            format_number(
                ai_intensity,
                2
            ),
            border=True,
        )

    else:

        st.metric(
            "AI Intensity",
            "N/A",
            border=True,
        )

with a4:

    if ai_pressure is not None:

        st.metric(
            "AI Central Pressure",
            f"{format_number(ai_pressure, 2)} hPa",
            border=True,
        )

    else:

        st.metric(
            "AI Central Pressure",
            "N/A",
            border=True,
        )


# =========================================================
# AI STATUS
# =========================================================

st.info(
    "**AI Analysis Status — Prototype Model**\n\n"
    "The multi-task CNN is generating experimental "
    "classification, intensity, and central-pressure "
    "outputs from the available multi-source satellite "
    "observations. The current demonstration dataset is "
    "small, so these outputs are not validated for "
    "operational forecasting.",
    icon="🤖",
)


# =========================================================
# CLASS PROBABILITIES
# =========================================================

if development_probabilities is not None:

    st.markdown(
        "#### Development Class Probabilities"
    )

    probability_cols = st.columns(
        len(
            development_probabilities
        )
    )

    for i, probability in enumerate(
        development_probabilities
    ):

        class_name = labels.get(
            str(i),
            "Unknown"
        )

        probability_value = float(
            probability
        )

        with probability_cols[i]:

            st.metric(
                class_name,
                f"{probability_value * 100:.1f}%",
                border=True,
            )

            st.progress(
                probability_value
            )


# =========================================================
# MULTI-SOURCE SATELLITE DATA
# =========================================================

st.divider()

st.subheader(
    "🛰️ Multi-Source Satellite Data"
)

left, right = st.columns(2)


# =========================================================
# INFRARED
# =========================================================

with left:

    st.markdown(
        "### 🌡️ Infrared Satellite"
    )

    ir_image = X[
        selected_index,
        :,
        :,
        0
    ]

    st.image(
        ir_image,
        caption=(
            "Normalized infrared "
            "brightness-temperature field"
        ),
        width="stretch",
    )

    st.caption(
        "Infrared observations provide thermal "
        "structure information useful for identifying "
        "cyclone cloud patterns."
    )


# =========================================================
# MICROWAVE
# =========================================================

with right:

    st.markdown(
        "### 📡 Passive Microwave"
    )

    availability_flags = record.get(
        "microwave_availability",
        []
    )

    microwave_channels_available = [
        channel
        for channel, flag in zip(
            channels[1:],
            availability_flags
        )
        if int(flag) == 1
    ]

    if microwave_channels_available:

        selected_channel = st.selectbox(
            "Select available microwave channel",
            microwave_channels_available,
        )

        channel_index = channels.index(
            selected_channel
        )

        microwave_image = X[
            selected_index,
            :,
            :,
            channel_index
        ]

        st.image(
            microwave_image,
            caption=selected_channel,
            width="stretch",
        )

        st.caption(
            "Passive microwave observations provide "
            "complementary information about cyclone "
            "structure."
        )

    else:

        st.warning(
            "No passive-microwave observation is "
            "available for this satellite observation.",
            icon="📡",
        )

        st.info(
            "The preprocessing pipeline retains the "
            "fixed 13-channel input shape using "
            "unavailable-channel flags.",
            icon="ℹ️",
        )


# =========================================================
# CYCLONE THERMAL STRUCTURE
# =========================================================

st.divider()

st.subheader(
    "🌪️ Cyclone Thermal Signal Map"
)

st.caption(
    "Prototype visualization of cyclone thermal "
    "structure derived from infrared satellite "
    "observations. This is not an AI forecast or "
    "operational prediction."
)

prediction_field = np.asarray(
    ir_image,
    dtype=np.float32
)

fig, ax = plt.subplots(
    figsize=(10, 5.5)
)

im = ax.imshow(
    prediction_field,
    cmap="turbo",
    origin="upper",
)

center_y, center_x = np.unravel_index(
    np.argmax(
        prediction_field
    ),
    prediction_field.shape
)

ax.scatter(
    center_x,
    center_y,
    s=180,
    facecolors="none",
    edgecolors="white",
    linewidths=2.5,
    label="Estimated signal center",
)

ax.set_title(
    f"Thermal Structure • "
    f"AI Class: {predicted_class}"
)

ax.set_xlabel(
    "Satellite X coordinate"
)

ax.set_ylabel(
    "Satellite Y coordinate"
)

cbar = fig.colorbar(
    im,
    ax=ax,
    pad=0.02,
)

cbar.set_label(
    "Normalized infrared signal"
)

ax.legend(
    loc="upper right"
)

fig.tight_layout()

st.pyplot(
    fig,
    width="stretch"
)

plt.close(fig)

st.info(
    "Color scale represents the normalized infrared "
    "field. The highlighted point is an image-derived "
    "signal location, not a validated forecast center.",
    icon="🌡️",
)


# =========================================================
# CYCLONE LOCATION
# =========================================================

st.divider()

st.subheader(
    "📍 Cyclone Location & Movement"
)

location_col, movement_col = st.columns(
    [1.2, 1]
)


# =========================================================
# LOCATION MAP
# =========================================================

with location_col:

    st.markdown(
        "### Geographic Observation"
    )

    try:

        lat_value = float(
            latitude
        )

        lon_value = float(
            longitude
        )

        map_data = pd.DataFrame(
            {
                "latitude": [lat_value],
                "longitude": [lon_value],
            }
        )

        st.map(
            map_data,
            latitude="latitude",
            longitude="longitude",
            zoom=4,
            height=380,
        )

    except Exception:

        st.warning(
            "Latitude/longitude values are not "
            "available for this observation.",
            icon="📍",
        )


# =========================================================
# MOVEMENT PARAMETERS
# =========================================================

with movement_col:

    st.markdown(
        "### Storm Motion"
    )

    storm_speed = get_value(
        record,
        "storm_speed"
    )

    storm_heading = get_value(
        record,
        "storm_heading"
    )

    meridional = get_value(
        record,
        "storm_speed_meridional_component"
    )

    zonal = get_value(
        record,
        "storm_speed_zonal_component"
    )

    m1, m2 = st.columns(2)

    with m1:

        st.metric(
            "Storm Speed",
            format_number(
                storm_speed,
                2
            ),
            border=True,
        )

        st.metric(
            "Meridional Component",
            format_number(
                meridional,
                2
            ),
            border=True,
        )

    with m2:

        st.metric(
            "Storm Heading",
            f"{format_number(storm_heading, 2)}°",
            border=True,
        )

        st.metric(
            "Zonal Component",
            format_number(
                zonal,
                2
            ),
            border=True,
        )

    st.info(
        "Movement parameters are derived from the "
        "available observation metadata and are "
        "displayed as environmental storm-motion "
        "information.",
        icon="🌪️",
    )


# =========================================================
# DATA QUALITY
# =========================================================

st.divider()

st.subheader(
    "🛰️ Data Quality & Sensor Availability"
)

q1, q2, q3, q4 = st.columns(4)

with q1:

    st.metric(
        "Observations Loaded",
        sample_count,
        border=True,
    )

with q2:

    st.metric(
        "Input Channels",
        X.shape[-1],
        border=True,
    )

with q3:

    finite_percentage = (
        np.isfinite(
            X[selected_index]
        ).mean()
        * 100
    )

    st.metric(
        "Finite Input Data",
        f"{finite_percentage:.1f}%",
        border=True,
    )

with q4:

    microwave_available_count = sum(
        int(x)
        for x in availability_flags
    )

    microwave_total_count = len(
        availability_flags
    )

    st.metric(
        "Microwave Coverage",
        f"{microwave_available_count}/"
        f"{microwave_total_count}",
        border=True,
    )


if microwave_channels_available:

    st.markdown(
        "**Available microwave channels:**"
    )

    channel_cols = st.columns(4)

    for i, channel in enumerate(
        microwave_channels_available
    ):

        with channel_cols[
            i % 4
        ]:

            st.success(
                channel
            )

else:

    st.warning(
        "No passive-microwave channels are "
        "available for this observation.",
        icon="📡",
    )


# =========================================================
# AI MODEL PIPELINE
# =========================================================

st.divider()

st.subheader(
    "🧠 AI Model Pipeline"
)

p1, p2, p3, p4 = st.columns(4)

with p1:

    st.markdown(
        """
        <div class="pipeline-box">
        <h4>🛰️ Multi-Source Input</h4>
        Infrared and passive microwave observations
        are converted into a common 128 × 128
        multi-channel input.
        </div>
        """,
        unsafe_allow_html=True,
    )

with p2:

    st.markdown(
        """
        <div class="pipeline-box">
        <h4>🧠 CNN Feature Extraction</h4>
        Convolutional layers extract spatial patterns
        from the multi-source satellite observations.
        </div>
        """,
        unsafe_allow_html=True,
    )

with p3:

    st.markdown(
        """
        <div class="pipeline-box">
        <h4>🌪️ Classification</h4>
        The development head estimates probabilities
        for the available cyclone development classes.
        </div>
        """,
        unsafe_allow_html=True,
    )

with p4:

    st.markdown(
        """
        <div class="pipeline-box">
        <h4>📊 Prediction</h4>
        Regression heads produce prototype estimates
        for cyclone intensity and central pressure.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# SYSTEM CAPABILITIES
# =========================================================

st.divider()

st.subheader(
    "🚀 System Capabilities"
)

cap1, cap2, cap3 = st.columns(3)

with cap1:

    st.markdown(
        """
        **Identification**

        - Satellite observation selection
        - Cyclone development information
        - Geographic location
        - Environmental parameters
        """
    )

with cap2:

    st.markdown(
        """
        **Classification**

        - CNN-based development classification
        - DB / TS probability distribution
        - Confidence visualization
        - Multi-source satellite input
        """
    )

with cap3:

    st.markdown(
        """
        **Prediction / Analysis**

        - Prototype intensity estimation
        - Prototype pressure estimation
        - Thermal structure visualization
        - Storm movement analysis
        """
    )


# =========================================================
# MODEL STATUS
# =========================================================

st.divider()

st.subheader(
    "🔬 Model Readiness"
)

if model is not None:

    st.success(
        "Multi-task CNN model loaded successfully.",
        icon="🧠",
    )

    st.info(
        "Model outputs: development classification, "
        "intensity regression, and central-pressure "
        "regression.",
        icon="📊",
    )

    st.info(
        "**Prototype Validation Status**\n\n"
        "The current demonstration uses 6 real satellite "
        "observations. The multi-task CNN architecture "
        "is operationally loaded, but the available "
        "dataset is not sufficient for meaningful model "
        "validation. Further training with a larger "
        "cyclone dataset is required before operational use.",
        icon="🔬",
    )

else:

    st.error(
        "CNN model file was not found.",
        icon="🚨",
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "CodeNova • Smart India Hackathon 2026 • "
    "Problem Statement 26070 • Disaster Management"
)

