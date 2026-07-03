import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import hashlib

# Load your trained model from the notebook
# For this example, we'll create a simple version
print("=" * 60)
print("DETECTING ATTACKS WITH LSTM AUTOENCODER")
print("=" * 60)

def simple_anomaly_score(data_chunk, expected_behavior='normal'):
    """Simplified anomaly detection:
    - Replay: repeated patterns
    - Injection: extreme values
    - Freeze: zero variance
    """    
    scores = {}
    
    # Check for variance (freeze has near-zero variance)
    temp_variance = np.var(data_chunk['temperature_c'].values)
    hum_variance = np.var(data_chunk['humidity_percent'].values)
    
    scores['freeze_score'] = 1.0 if (temp_variance < 0.01 or hum_variance < 0.01) else 0.0
    
    # Check for extreme values (injection)
    temp_mean = data_chunk['temperature_c'].mean()
    temp_std = data_chunk['temperature_c'].std()
    extreme_count = sum(1 for t in data_chunk['temperature_c'] 
                       if abs(t - temp_mean) > 3 * (temp_std + 0.1))
    
    scores['injection_score'] = min(extreme_count / len(data_chunk), 1.0)
    
    # Check for repeated patterns (replay)
    temp_diffs = np.diff(data_chunk['temperature_c'].values)
    zero_diffs = sum(1 for d in temp_diffs if abs(d) < 0.01)
    
    scores['replay_score'] = min(zero_diffs / len(temp_diffs), 1.0)
    
    return scores

# Analyze each attack
attacks = {
    'replay': 'replay_attack.csv',
    'injection': 'injection_attack.csv',
    'freeze': 'freeze_attack.csv'
}

for attack_name, filename in attacks.items():
    print(f"\n{'=' * 60}")
    print(f"📊 Analyzing: {attack_name.upper()} ATTACK")
    print(f"{'=' * 60}")
    
df = pd.read_csv(filename)
    
    # Get last 100 readings (where attack likely is)
    attack_window = df.tail(100)
    
scores = simple_anomaly_score(attack_window)
    
    print(f"\nAnomaly Scores (0=normal, 1=highly anomalous):")
    print(f"  🔄 Replay Score:     {scores['replay_score']:.3f}")
    print(f"  🔌 Injection Score:  {scores['injection_score']:.3f}")
    print(f"  ❄️  Freeze Score:     {scores['freeze_score']:.3f}")
    
    # Determine primary attack
    max_score = max(scores.values())
    primary_attack = [k for k, v in scores.items() if v == max_score][0]
    
    print(f"\n✅ PRIMARY ATTACK DETECTED: {primary_attack.upper()}")
    print(f"   Confidence: {max_score*100:.1f}%")
    
    # Show sample data
    print(f"\n📈 Sample readings (last 5):")
    print(df.tail(5)[['timestamp', 'temperature_c', 'humidity_percent']].to_string())

print(f"\n{'=' * 60}")
print("✅ DETECTION COMPLETE")
print(f"{'=' * 60}")
