import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ==============================
# FILE PATHS
# ==============================
FILES = [
    ("raw_data/Train_Test_IoT_Weather.csv", "sensor_1"),
    ("raw_data/dht11_dataset_10000.csv", "sensor_2"),
    ("raw_data/iot_sensor_dataset.csv", "sensor_3"),
    ("raw_data/log_temp.csv", "sensor_4")
]

TIME_INTERVAL = "10s"

all_data = []

# ==============================
# CLEAN FUNCTION
# ==============================
def clean_dataset(file_path, sensor_id):
    print(f"\nProcessing: {file_path}")

    try:
        df = pd.read_csv(file_path, low_memory=False)
    except:
        print(f"❌ Could not read {file_path}")
        return None

    df = df.copy()

    # Standardize columns
    df.columns = df.columns.str.strip().str.lower()

    rename_map = {}
    for col in df.columns:
        c = col.lower()
        if c in ["ts", "time", "date", "timestamp"]:
            rename_map[col] = "timestamp"
        elif c in ["temp", "temperature", "temperature_c"]:
            rename_map[col] = "temperature"
        elif c in ["humidity", "humidity_percent"]:
            rename_map[col] = "humidity"

    df.rename(columns=rename_map, inplace=True)

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Ensure required columns
    if not all(col in df.columns for col in ["timestamp", "temperature", "humidity"]):
        print(f"⚠️ Skipping {file_path}")
        return None

    df = df[["timestamp", "temperature", "humidity"]]

    # Timestamp parsing
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")

    # Numeric conversion
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
    df = df.dropna()

    # Range filtering
    df = df[
        (df["temperature"].between(-50, 100)) &
        (df["humidity"].between(0, 100))
    ]

    if len(df) == 0:
        print(f"⚠️ Empty after filtering")
        return None

    # Set index
    df.set_index("timestamp", inplace=True)

    # Resample
    df = df.resample(TIME_INTERVAL).mean()

    # Interpolate
    df["temperature"] = df["temperature"].interpolate()
    df["humidity"] = df["humidity"].interpolate()

    df = df.dropna()

    if len(df) == 0:
        print(f"⚠️ Empty after resample")
        return None

    # Add sensor ID safely
    df = df.copy()
    df["sensor_id"] = sensor_id

    return df.reset_index()


# ==============================
# PROCESS ALL FILES
# ==============================
for file_path, sensor_id in FILES:
    cleaned = clean_dataset(file_path, sensor_id)
    if cleaned is not None:
        print(f"✅ {sensor_id} rows:", len(cleaned))
        all_data.append(cleaned.copy())

# ==============================
# MERGE DATA
# ==============================
final_df = pd.concat([df.copy() for df in all_data], ignore_index=True)

print("\nBefore alignment:")
print(final_df["sensor_id"].value_counts())

# ==============================
# FIX 1: PER-SENSOR TIME ALIGNMENT
# ==============================
aligned_list = []

for sensor in final_df["sensor_id"].unique():
    temp_df = final_df[final_df["sensor_id"] == sensor].copy()

    start_time = temp_df["timestamp"].max() - pd.Timedelta(hours=6)
    temp_df = temp_df[temp_df["timestamp"] >= start_time]

    aligned_list.append(temp_df)

final_df = pd.concat(aligned_list, ignore_index=True)

print("\nAfter alignment:")
print(final_df["sensor_id"].value_counts())

# ==============================
# FIX 2: BALANCE DATA
# ==============================
counts = final_df["sensor_id"].value_counts()

if counts.min() == 0:
    raise ValueError("❌ One sensor has no data after alignment")

min_size = counts.min()

balanced_list = []

for sensor in final_df["sensor_id"].unique():
    temp_df = final_df[final_df["sensor_id"] == sensor].copy()
    temp_df = temp_df.sample(min_size, random_state=42)
    balanced_list.append(temp_df)

final_df = pd.concat(balanced_list, ignore_index=True)

print("\nAfter balancing:")
print(final_df["sensor_id"].value_counts())

# ==============================
# SORT DATA
# ==============================
final_df = final_df.sort_values(["sensor_id", "timestamp"])

# ==============================
# FIX 3: NORMALIZE PER SENSOR
# ==============================
for sensor in final_df["sensor_id"].unique():
    mask = final_df["sensor_id"] == sensor

    scaler = MinMaxScaler()

    final_df.loc[mask, ["temperature", "humidity"]] = scaler.fit_transform(
        final_df.loc[mask, ["temperature", "humidity"]]
    )

# ==============================
# SAVE FINAL DATASET
# ==============================
final_df.to_csv("multi_sensor_cleaned_balanced.csv", index=False)

print("\n===================================")
print("✅ FINAL DATA READY")
print("===================================")
print("Shape:", final_df.shape)
print("\nSample:")
print(final_df.head())