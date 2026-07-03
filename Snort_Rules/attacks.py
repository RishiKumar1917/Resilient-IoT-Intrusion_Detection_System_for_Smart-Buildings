import pandas as pd
import numpy as np
import random

# Load your dataset
df = pd.read_csv('dht11_dataset_10000.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("=" * 60)
print("CREATING IoT SENSOR ATTACKS")
print("=" * 60)

# ============ ATTACK 1: REPLAY ATTACK ============
print("\n1️⃣  REPLAY ATTACK: Repeat normal readings out of order")
replay_df = df.copy()

# Take readings 100-150 and repeat them at the end (simulating attacker replaying old data)
replay_chunk = df.iloc[100:150].copy()
replay_chunk['timestamp'] = pd.date_range(
    start=df['timestamp'].max() + pd.Timedelta(seconds=1),
    periods=len(replay_chunk),
    freq='1S'
)

replay_df = pd.concat([replay_df, replay_chunk], ignore_index=True)
replay_df.to_csv('replay_attack.csv', index=False)

print(f"✓ Original: {len(df)} rows")
print(f"✓ With replay: {len(replay_df)} rows (+{len(replay_chunk)} replayed)")
print(f"\nReplayed chunk (temp/humidity):\n{replay_chunk[['timestamp', 'temperature_c', 'humidity_percent']].head()}")

# ============ ATTACK 2: DATA INJECTION ATTACK ============
print("\n" + "=" * 60)
print("2️⃣  DATA INJECTION: Insert false extreme readings")
inject_df = df.copy()

# Inject 50 fake extreme readings
n_inject = 50
inject_temps = np.random.uniform(55, 70, n_inject)  # Physically impossible range
inject_hums = np.random.uniform(5, 15, n_inject)    # Way too dry

for i in range(n_inject):
    new_row = {
        'timestamp': df['timestamp'].max() + pd.Timedelta(seconds=i),
        'temperature_c': round(inject_temps[i], 2),
        'humidity_percent': round(inject_hums[i], 2),
        'device_id': 'DHT11_01',
        'room_id': 'room_3',
        'status': 'anomaly'
    }
    inject_df = pd.concat([inject_df, pd.DataFrame([new_row])], ignore_index=True)

inject_df.to_csv('injection_attack.csv', index=False)

print(f"✓ Original: {len(df)} rows")
print(f"✓ With injection: {len(inject_df)} rows (+{n_inject} fake readings)")
print(f"\nInjected extreme readings:\n{inject_df.tail(10)[['timestamp', 'temperature_c', 'humidity_percent', 'status']]}\n")

# ============ ATTACK 3: SENSOR FREEZE/STUCK-AT ATTACK ============
print("\n" + "=" * 60)
print("3️⃣  SENSOR FREEZE: Readings stuck at one value")
freeze_df = df.copy()

# Last 100 readings freeze at a constant value
freeze_start = len(freeze_df) - 100
frozen_temp = df.iloc[-1]['temperature_c']
frozen_hum = df.iloc[-1]['humidity_percent']

for i in range(freeze_start, len(freeze_df)):
    freeze_df.at[i, 'temperature_c'] = frozen_temp
    freeze_df.at[i, 'humidity_percent'] = frozen_hum
    freeze_df.at[i, 'status'] = 'anomaly'

freeze_df.to_csv('freeze_attack.csv', index=False)

print(f"✓ Original: {len(df)} rows")
print(f"✓ Frozen readings: last 100 rows stuck at ({frozen_temp}°C, {frozen_hum}%)")
print(f"\nFrozen chunk (last 10 rows - all identical):\n{freeze_df.tail(10)[['timestamp', 'temperature_c', 'humidity_percent']]}\n")

print("\n" + "=" * 60)
print("✅ All attack datasets created!")
print("=" * 60)
