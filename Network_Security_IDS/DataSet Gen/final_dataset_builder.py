import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("../enhanced_iot_dataset.csv")

# Sort by time
df = df.sort_values(by="timestamp").reset_index(drop=True)

improved_df = df.copy()

window_size = 30

# -------------------------------
# 🔴 REPLAY ATTACK (REALISTIC + STRONG)
# -------------------------------
replay_indices = improved_df[improved_df["attack_type"] == "replay_attack"].index

for i in range(0, len(replay_indices), 80):

    segment = replay_indices[i:i+60]

    if len(segment) < window_size:
        continue

    start = segment[0]
    end = segment[-1]

    replay_block = improved_df.loc[start:end].copy()

    delay = np.random.randint(20, 50)
    insert_pos = min(end + delay, len(improved_df) - 60)

    for j in range(len(replay_block)):
        if insert_pos + j < len(improved_df):

            temp_val = replay_block.iloc[j]["temperature_c"] + np.random.normal(0, 0.2)
            hum_val = replay_block.iloc[j]["humidity_percent"] + np.random.normal(0, 0.3)

            improved_df.loc[insert_pos + j, "temperature_c"] = temp_val
            improved_df.loc[insert_pos + j, "humidity_percent"] = hum_val
            improved_df.loc[insert_pos + j, "attack_type"] = "replay_attack"

# -------------------------------
# 🟠 DRIFT ATTACK (GRADUAL TREND)
# -------------------------------
drift_indices = improved_df[improved_df["attack_type"] == "drift_attack"].index

for idx in drift_indices:

    base_temp = improved_df.loc[idx, "temperature_c"]
    base_hum = improved_df.loc[idx, "humidity_percent"]

    for k in range(25):
        if idx + k < len(improved_df):
            improved_df.loc[idx + k, "temperature_c"] = base_temp + (0.03 * k)
            improved_df.loc[idx + k, "humidity_percent"] = base_hum + (0.02 * k)

# -------------------------------
# 🟡 NOISE ATTACK (BURST NOISE)
# -------------------------------
noise_indices = improved_df[improved_df["attack_type"] == "noise_attack"].index

for idx in noise_indices:

    for k in range(np.random.randint(3, 8)):
        if idx + k < len(improved_df):
            improved_df.loc[idx + k, "temperature_c"] += np.random.normal(0, 1.2)
            improved_df.loc[idx + k, "humidity_percent"] += np.random.normal(0, 1.5)

# -------------------------------
# 🔵 INJECTION ATTACK (SHORT SPIKE BURST)
# -------------------------------
inj_indices = improved_df[improved_df["attack_type"] == "injection_attack"].index

for idx in inj_indices:

    spike = np.random.choice([4, -4, 5, -5])

    for k in range(np.random.randint(2, 5)):
        if idx + k < len(improved_df):
            improved_df.loc[idx + k, "temperature_c"] += spike
            improved_df.loc[idx + k, "humidity_percent"] += spike * 0.7

# -------------------------------
# 🟣 DROP ATTACK (SENSOR FREEZE)
# -------------------------------
drop_indices = improved_df[improved_df["attack_type"] == "drop_attack"].index

for idx in drop_indices:

    freeze_temp = improved_df.loc[idx, "temperature_c"]
    freeze_hum = improved_df.loc[idx, "humidity_percent"]

    for k in range(np.random.randint(5, 12)):
        if idx + k < len(improved_df):
            improved_df.loc[idx + k, "temperature_c"] = freeze_temp + np.random.uniform(-0.3, 0.3)
            improved_df.loc[idx + k, "humidity_percent"] = freeze_hum + np.random.uniform(-0.5, 0.5)

# -------------------------------
# 🟢 CLIP TO REALISTIC SENSOR RANGES
# -------------------------------
improved_df["temperature_c"] = improved_df["temperature_c"].clip(10, 45)
improved_df["humidity_percent"] = improved_df["humidity_percent"].clip(20, 85)

# -------------------------------
# 🟢 BALANCE DATASET
# -------------------------------
balanced_df = improved_df.groupby("attack_type").apply(
    lambda x: x.sample(n=800, replace=True)
).reset_index(drop=True)

# Shuffle dataset
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

# Save final dataset
balanced_df.to_csv("enhanced_iot_dataset_final.csv", index=False)

print("✅ FINAL dataset saved as enhanced_iot_dataset_final.csv")