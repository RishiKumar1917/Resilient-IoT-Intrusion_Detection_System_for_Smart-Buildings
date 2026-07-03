import pandas as pd
import numpy as np

np.random.seed(42)

NUM_SENSORS = 3
ROWS_PER_SENSOR = 1000

data = []

# -------------------------------
# 🟢 BASE NORMAL DATA
# -------------------------------
start_time = pd.Timestamp("2025-01-01 00:00:00")

base_temp_map = {}
base_hum_map = {}

# initialize sensor baselines
for sensor in range(NUM_SENSORS):
    base_temp_map[sensor] = 25 + np.random.uniform(-2, 2)
    base_hum_map[sensor] = 50 + np.random.uniform(-5, 5)

# interleaved generation
for t in range(ROWS_PER_SENSOR):
    for sensor in range(NUM_SENSORS):
        timestamp = start_time + pd.Timedelta(seconds=(t * NUM_SENSORS + sensor))

        # CHANGE 7: Tightened normal noise for cleaner baseline
        temp = base_temp_map[sensor] + np.random.normal(0, 0.3)   # was 0.5
        hum  = base_hum_map[sensor]  + np.random.normal(0, 0.6)   # was 1.0

        data.append(
            {
                "timestamp": timestamp,
                "step": t,
                "sensor_id": f"sensor_{sensor}",
                "temperature_c": temp,
                "humidity_percent": hum,
                "attack_type": "normal",
            }
        )

df = pd.DataFrame(data)

# -------------------------------
# 🟠 DRIFT ATTACK
# -------------------------------
drift_sensors = ["sensor_0"]
drift_ranges = {}

for sensor in drift_sensors:

    # CHANGE 4: Extended window (was 200–300)
    start, end = 200, 350
    drift_ranges[sensor] = (start, end)

    base_vals = df[
        (df.sensor_id == sensor) &
        (df.step == start)
    ][["temperature_c", "humidity_percent"]].values[0]

    for i, t in enumerate(range(start, end)):

        drift_factor = (i / 100) ** 1.5

        # CHANGE 4: Stronger drift magnitude (was 8 / 10)
        temp = base_vals[0] + drift_factor * 15 + np.random.normal(0, 0.3)
        hum  = base_vals[1] + drift_factor * 10 + np.random.normal(0, 0.5)

        df.loc[
            (df.sensor_id == sensor) & (df.step == t),
            ["temperature_c", "humidity_percent", "attack_type"]
        ] = [temp, hum, "drift_attack"]

print("\n🟠 Drift Attack Locations:", drift_ranges)

# -------------------------------
# 🔵 INJECTION ATTACK
# -------------------------------
injection_ranges = {}

# Burst
sensor = "sensor_1"
start, end = 300, 340
injection_ranges[sensor] = (start, end)

for t in range(start, end):
    # CHANGE 1: Stronger injection spike (was +6 ± 1)
    df.loc[(df.sensor_id == sensor) & (df.step == t), "temperature_c"] += 10 + np.random.normal(0, 2)
    df.loc[(df.sensor_id == sensor) & (df.step == t), "attack_type"] = "injection_attack"

# Periodic
sensor = "sensor_1"
start, end = 600, 660
injection_ranges[sensor] = (start, end)

for t in range(start, end):
    if t % 5 == 0:
        # CHANGE 1: Stronger injection spike (was +7 ± 1)
        df.loc[(df.sensor_id == sensor) & (df.step == t), "temperature_c"] += 10 + np.random.normal(0, 2)
        df.loc[(df.sensor_id == sensor) & (df.step == t), "attack_type"] = "injection_attack"

# Negative
sensor = "sensor_1"
start, end = 700, 740
injection_ranges[sensor] = (start, end)

for t in range(start, end):
    # CHANGE 1: Stronger negative injection (was -6 ± 1)
    df.loc[(df.sensor_id == sensor) & (df.step == t), "temperature_c"] += -10 + np.random.normal(0, 2)
    df.loc[(df.sensor_id == sensor) & (df.step == t), "attack_type"] = "injection_attack"

print("\n🔵 Injection Attack Locations:", injection_ranges)

# -------------------------------
# 🟣 DROP ATTACK
# -------------------------------
drop_ranges = {}

# Soft freeze
sensor = "sensor_2"
start, end = 400, 460
drop_ranges[sensor] = (start, end)

base_val = df.loc[(df.sensor_id == sensor) & (df.step == start), "temperature_c"].values[0]

for t in range(start, end):
    df.loc[(df.sensor_id == sensor) & (df.step == t), "temperature_c"] = base_val + np.random.normal(0, 0.05)
    df.loc[(df.sensor_id == sensor) & (df.step == t), "attack_type"] = "drop_attack"

# Step drop
sensor = "sensor_2"
start, end = 600, 660
drop_ranges[sensor] = (start, end)

for t in range(start, end):
    # CHANGE 2: Larger initial step drop (was -8) and sustained drop (was -6 ± 0.2)
    if t < start + 5:
        df.loc[(df.sensor_id == sensor) & (df.step == t), "temperature_c"] -= 12
    else:
        df.loc[(df.sensor_id == sensor) & (df.step == t), "temperature_c"] -= 9 + np.random.normal(0, 0.5)

    df.loc[(df.sensor_id == sensor) & (df.step == t), "attack_type"] = "drop_attack"

# Intermittent
sensor = "sensor_2"
start, end = 750, 800
drop_ranges[sensor] = (start, end)

freeze_val = df.loc[(df.sensor_id == sensor) & (df.step == start), "temperature_c"].values[0]

for t in range(start, end):
    if (t // 5) % 2 == 0:
        val = freeze_val + np.random.normal(0, 0.1)
    else:
        val = freeze_val + np.random.normal(0, 1)

    df.loc[(df.sensor_id == sensor) & (df.step == t), "temperature_c"] = val
    df.loc[(df.sensor_id == sensor) & (df.step == t), "attack_type"] = "drop_attack"

print("\n🟣 Drop Attack Locations:", drop_ranges)

# -------------------------------
# 🟡 NOISE ATTACK
# -------------------------------
noise_ranges = {}

# High freq
sensor = "sensor_1"
start, end = 500, 560
noise_ranges[sensor] = (start, end)

for t in range(start, end):
    # CHANGE 3: Much higher noise std (was 1.5)
    df.loc[(df.sensor_id == sensor) & (df.step == t), "temperature_c"] += np.random.normal(0, 3.5)
    df.loc[(df.sensor_id == sensor) & (df.step == t), "attack_type"] = "noise_attack"

# Wave
sensor = "sensor_1"
start, end = 800, 860
noise_ranges[sensor] = (start, end)

for i, t in enumerate(range(start, end)):
    df.loc[(df.sensor_id == sensor) & (df.step == t), "temperature_c"] += 2 * np.sin(i / 3) + np.random.normal(0, 0.5)
    df.loc[(df.sensor_id == sensor) & (df.step == t), "attack_type"] = "noise_attack"

# Mixed
sensor = "sensor_1"
start, end = 650, 700
noise_ranges[sensor] = (start, end)

for t in range(start, end):
    # CHANGE 3: Stronger mixed noise (was 2.0 / 0.7)
    val = np.random.normal(0, 3.0) if t % 4 == 0 else np.random.normal(0, 1.5)
    df.loc[(df.sensor_id == sensor) & (df.step == t), "temperature_c"] += val
    df.loc[(df.sensor_id == sensor) & (df.step == t), "attack_type"] = "noise_attack"

print("\n🟡 Noise Attack Locations:", noise_ranges)

# -------------------------------
# 🔴 REPLAY ATTACK (LAST!)
# -------------------------------
for sensor in ["sensor_0", "sensor_2"]:

    block = df[(df.sensor_id == sensor) & (df.step.between(100, 130))].copy()

    for _, row in block.iterrows():

        new_t = row["step"] + 200

        # CHANGE 5: Exact replay — no noise added
        df.loc[
            (df.sensor_id == sensor) & (df.step == new_t),
            ["temperature_c", "humidity_percent", "attack_type"]
        ] = [
            row["temperature_c"],
            row["humidity_percent"],
            "replay_attack"
        ]

# Cross sensor replay
block = df[(df.sensor_id == "sensor_0") & (df.step.between(150, 180))].copy()

for _, row in block.iterrows():

    new_t = row["step"] + 250

    # CHANGE 5: Exact replay — no noise added
    df.loc[
        (df.sensor_id == "sensor_2") & (df.step == new_t),
        ["temperature_c", "humidity_percent", "attack_type"]
    ] = [
        row["temperature_c"],
        row["humidity_percent"],
        "replay_attack"
    ]

# CHANGE 5: Hard replay — exact repeating block on sensor_2
for sensor in ["sensor_0"]:
    block = df[(df.sensor_id == sensor) & (df.step.between(50, 80))].copy()

    for _, row in block.iterrows():
        new_t = row["step"] + 300

        df.loc[
            (df.sensor_id == sensor) & (df.step == new_t),
            ["temperature_c", "humidity_percent", "attack_type"]
        ] = [
            row["temperature_c"],
            row["humidity_percent"],
            "replay_attack"
        ]

print("\n🔴 Replay Attack Added")

print("\n🔴 Replay Attack Locations:")
print("sensor_0 → 300–330, 400–430")
print("sensor_2 → 300–330, 400–430")
print("sensor_2 → 400–430 (cross-sensor)")
print("sensor_0 → 350–380 (hard replay from steps 50–80)")

# -------------------------------
# 🟢 HUMIDITY DISTURBANCE FOR ALL ATTACKS (CHANGE 6)
# -------------------------------
attack_mask = df["attack_type"] != "normal"
df.loc[attack_mask, "humidity_percent"] += np.random.uniform(-8, 8, size=attack_mask.sum())

# -------------------------------
# 🟢 CLIP
# -------------------------------
df["temperature_c"] = df["temperature_c"].clip(15, 45)
df["humidity_percent"] = df["humidity_percent"].clip(20, 80)

# -------------------------------
# SAVE
# -------------------------------
df = df.drop(columns=["step"])
df = df.sort_values("timestamp").reset_index(drop=True)

df.to_csv("enhanced_iot_dataset_3sensors.csv", index=False)

print("\nSensors:", df["sensor_id"].unique())
print("Class counts:", df["attack_type"].value_counts())

print("\n✅ FINAL DATASET GENERATED")