import os
import glob
import json

import numpy as np
import xarray as xr
from scipy.ndimage import zoom


# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = os.path.join("dataset", "raw", "IO03")
OUTPUT_DIR = "data"

IMAGE_SIZE = 128

MICROWAVE_CHANNELS = [
    ("S1", "TB_10.65V"),
    ("S1", "TB_10.65H"),
    ("S2", "TB_18.7V"),
    ("S2", "TB_18.7H"),
    ("S3", "TB_23.8V"),
    ("S3", "TB_23.8H"),
    ("S4", "TB_36.5V"),
    ("S4", "TB_36.5H"),
    ("S5", "TB_A89.0V"),
    ("S5", "TB_A89.0H"),
    ("S6", "TB_B89.0V"),
    ("S6", "TB_B89.0H"),
]


# ============================================================
# IMAGE HELPERS
# ============================================================

def resize_2d(array, size=128):
    """Resize a 2D array to a fixed square size."""

    array = np.asarray(array, dtype=np.float32)

    if array.ndim != 2:
        raise ValueError(
            f"Expected 2D array, got shape {array.shape}"
        )

    zoom_y = size / array.shape[0]
    zoom_x = size / array.shape[1]

    resized = zoom(
        array,
        (zoom_y, zoom_x),
        order=1
    )

    return resized.astype(np.float32)


def normalize_channel(array):
    """Normalize one channel to the range 0-1."""

    array = np.asarray(array, dtype=np.float32)

    finite = np.isfinite(array)

    if not np.any(finite):
        return np.zeros_like(
            array,
            dtype=np.float32
        )

    minimum = np.nanmin(array)
    maximum = np.nanmax(array)

    array = np.nan_to_num(
        array,
        nan=float(minimum),
        posinf=float(maximum),
        neginf=float(minimum)
    )

    if maximum - minimum < 1e-8:
        return np.zeros_like(
            array,
            dtype=np.float32
        )

    array = (
        (array - minimum)
        / (maximum - minimum)
    )

    return np.clip(
        array,
        0.0,
        1.0
    ).astype(np.float32)


# ============================================================
# METADATA HELPER
# ============================================================

def read_scalar(group, variable, default=np.nan):
    """Safely read one metadata value, including NetCDF strings."""

    try:
        value = group[variable].values
        value = np.asarray(value).flatten()

        if len(value) == 0:
            return default

        value = value[0]

        if isinstance(value, bytes):
            return value.decode("utf-8").strip()

        if isinstance(value, str):
            return value.strip()

        if hasattr(value, "item"):
            value = value.item()

        return value

    except Exception:
        return default


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(filepath):

    print()
    print("=" * 70)
    print("Processing:", os.path.basename(filepath))
    print("=" * 70)

    # --------------------------------------------------------
    # 1. INFRARED CHANNEL
    # --------------------------------------------------------

    infrared = xr.open_dataset(
        filepath,
        group="infrared",
        engine="netcdf4"
    )

    ir = infrared["IRWIN"].values

    print("IR shape:", ir.shape)

    ir = resize_2d(
        ir,
        IMAGE_SIZE
    )

    ir = normalize_channel(ir)

    infrared.close()

    channels = [ir]

    channel_names = ["IRWIN"]

    # --------------------------------------------------------
    # 2. PASSIVE MICROWAVE CHANNELS
    # --------------------------------------------------------

    availability = []

    for subgroup, variable in MICROWAVE_CHANNELS:

        try:

            ds = xr.open_dataset(
                filepath,
                group=f"passive_microwave/{subgroup}",
                engine="netcdf4"
            )

            if variable not in ds:

                print(
                    "Missing:",
                    variable,
                    "-> using zero-filled channel"
                )

                data = np.zeros(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    dtype=np.float32
                )

                availability.append(0)

            else:

                data = ds[variable].values

                print(
                    f"Microwave {variable}: "
                    f"{data.shape}"
                )

                data = resize_2d(
                    data,
                    IMAGE_SIZE
                )

                data = normalize_channel(data)

                availability.append(1)

            channels.append(data)
            channel_names.append(variable)

            ds.close()

        except Exception as error:

            print(
                f"Unavailable {subgroup}/{variable}: "
                f"{error}"
            )

            data = np.zeros(
                (IMAGE_SIZE, IMAGE_SIZE),
                dtype=np.float32
            )

            channels.append(data)
            channel_names.append(variable)

            availability.append(0)

    # --------------------------------------------------------
    # 3. STACK TO FIXED 13-CHANNEL INPUT
    # --------------------------------------------------------

    image = np.stack(
        channels,
        axis=-1
    )

    print(
        "Final image shape:",
        image.shape
    )

    print(
        "Channels:",
        channel_names
    )

    print(
        "Microwave availability:",
        availability
    )

    # --------------------------------------------------------
    # 4. OVERPASS SENSOR METADATA
    # --------------------------------------------------------
    overpass = xr.open_dataset(
        filepath,
        group="overpass_metadata",
        engine="netcdf4"
    )

    instrument_name = read_scalar(
        overpass,
        "instrument_name",
        default="UNKNOWN"
    )

    platform_name = read_scalar(
        overpass,
        "platform_name",
        default="UNKNOWN"
    )

    overpass.close()

    # 5. STORM METADATA
    # --------------------------------------------------------

    storm = xr.open_dataset(
        filepath,
        group="overpass_storm_metadata",
        engine="netcdf4"
    )

    intensity = read_scalar(
        storm,
        "intensity"
    )

    pressure = read_scalar(
        storm,
        "central_min_pressure"
    )

    latitude = read_scalar(
        storm,
        "storm_latitude"
    )

    longitude = read_scalar(
        storm,
        "storm_longitude"
    )

    storm_speed = read_scalar(
        storm,
        "storm_speed"
    )

    storm_heading = read_scalar(
        storm,
        "storm_heading"
    )

    storm_speed_meridional_component = read_scalar(
        storm,
        "storm_speed_meridional_component"
    )

    storm_speed_zonal_component = read_scalar(
        storm,
        "storm_speed_zonal_component"
    )

    development = read_scalar(
        storm,
        "development_level",
        default="UNKNOWN"
    )

    metadata = {
        "filename": os.path.basename(filepath),
        "instrument_name": str(instrument_name),
        "platform_name": str(platform_name),
        "development_level": str(development),
        "intensity": float(intensity) if np.isfinite(intensity) else None,
        "central_min_pressure": float(pressure) if np.isfinite(pressure) else None,
        "storm_latitude": float(latitude) if np.isfinite(latitude) else None,
        "storm_longitude": float(longitude) if np.isfinite(longitude) else None,
        "storm_speed": float(storm_speed) if np.isfinite(storm_speed) else None,
        "storm_heading": float(storm_heading) if np.isfinite(storm_heading) else None,
        "storm_speed_meridional_component": float(storm_speed_meridional_component) if np.isfinite(storm_speed_meridional_component) else None,
        "storm_speed_zonal_component": float(storm_speed_zonal_component) if np.isfinite(storm_speed_zonal_component) else None,
        "microwave_availability": availability,
    }

    storm.close()

    return image, metadata, channel_names


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    files = sorted(
        glob.glob(
            os.path.join(
                RAW_DIR,
                "*.nc"
            )
        )
    )

    print()
    print("NOAA TC PRIMED DATASET EXTRACTOR")
    print("=" * 70)
    print("Input directory:", RAW_DIR)
    print("NetCDF files found:", len(files))
    print("=" * 70)

    if not files:
        raise FileNotFoundError(
            f"No NetCDF files found in {RAW_DIR}"
        )

    images = []
    metadata = []
    channel_names = None

    for filepath in files:

        try:

            image, info, names = process_file(
                filepath
            )

            images.append(image)
            metadata.append(info)

            if channel_names is None:
                channel_names = names

        except Exception as error:

            print()
            print("ERROR processing:")
            print(filepath)
            print(error)
            print("Skipping this file.")

    if not images:
        raise RuntimeError(
            "No files were successfully processed."
        )

    X = np.stack(
        images,
        axis=0
    )

    print()
    print("=" * 70)
    print("DATASET CREATED")
    print("=" * 70)
    print("X shape:", X.shape)
    print("X dtype:", X.dtype)
    print("Number of samples:", len(X))
    print("Number of channels:", X.shape[-1])

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "X_multisource.npy"
        ),
        X
    )

    with open(
        os.path.join(
            OUTPUT_DIR,
            "metadata.json"
        ),
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )

    with open(
        os.path.join(
            OUTPUT_DIR,
            "channels.json"
        ),
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            channel_names,
            file,
            indent=2
        )

    print()
    print("Saved:")
    print("  data/X_multisource.npy")
    print("  data/metadata.json")
    print("  data/channels.json")
    print()
    print("Extraction complete.")


if __name__ == "__main__":
    main()
