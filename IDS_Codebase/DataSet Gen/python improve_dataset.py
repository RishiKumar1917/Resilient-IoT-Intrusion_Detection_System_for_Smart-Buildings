import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("../enhanced_iot_dataset.csv")

# Ensure sorting by time
df = df.sort_values(by="timestamp").reset_index(drop=True)

# Copy dataset
improved_df = df.copy()

# -------------------------------
# 🔴 1. REPLAY ATTACK IMPROVEMENT
# -------------------------------
replay_indices = improved_df[improved_df["attack_type"] == "replay_attack"].index

for i in range(0, len(replay_indices), 40):
    segment = replay_indices[i:i+40]

    if len(segment) < 20:
        continue

    start = segment[0]
    end = segment[-1]

    # Copy segment and repeat it
    replay_block = improved_df.loc[start:end].copy()

    insert_pos = min(end + 5, len(improved_df)-1)

    for j in range(len(replay_block)):
        if insert_pos + j < len(improved_df):
            improved_df.loc[insert_pos + j, "temperature_c"] = replay_block.iloc[j]["temperature_c"]
            improved_df.loc[insert_pos + j, "humidity_percent"] = replay_block.iloc[j]["humidity_percent"]
            improved_df.loc[insert_pos + j, "attack_type"] = "replay_attack"

# -------------------------------
# 🟠 2. DRIFT ATTACK IMPROVEMENT
# -------------------------------
drift_indices = improved_df[improved_df["attack_type"] == "drift_attack"].index

for idx in drift_indices:
    for k in range(10):
        if idx + k < len(improved_df):
            improved_df.loc[idx + k, "temperature_c"] += 0.05 * k
            improved_df.loc[idx + k, "humidity_percent"] += 0.03 * k

# -------------------------------
# 🟡 3. NOISE ATTACK IMPROVEMENT
# -------------------------------
noise_indices = improved_df[improved_df["attack_type"] == "noise_attack"].index

for idx in noise_indices:
    improved_df.loc[idx, "temperature_c"] += np.random.normal(0, 1.5)
    improved_df.loc[idx, "humidity_percent"] += np.random.normal(0, 1.5)

# -------------------------------
# 🔵 4. INJECTION ATTACK IMPROVEMENT
# -------------------------------
inj_indices = improved_df[improved_df["attack_type"] == "injection_attack"].index

for idx in inj_indices:
    spike = np.random.choice([6, -6, 8, -8])
    improved_df.loc[idx, "temperature_c"] += spike
    improved_df.loc[idx, "humidity_percent"] += spike / 2

# -------------------------------
# 🟣 5. DROP ATTACK IMPROVEMENT
# -------------------------------
drop_indices = improved_df[improved_df["attack_type"] == "drop_attack"].index

for idx in drop_indices:
    for k in range(5):
        if idx + k < len(improved_df):
            improved_df.loc[idx + k, "temperature_c"] = 0
            improved_df.loc[idx + k, "humidity_percent"] = 0

# -------------------------------
# 🟢 6. BALANCE DATASET
# -------------------------------
balanced_df = improved_df.groupby("attack_type").apply(
    lambda x: x.sample(n=800, replace=True)
).reset_index(drop=True)

# Shuffle
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

# Save new dataset
balanced_df.to_csv("enhanced_iot_dataset_improved.csv", index=False)

print("✅ Improved dataset saved as enhanced_iot_dataset_improved.csv")