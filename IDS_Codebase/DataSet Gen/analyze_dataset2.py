import pandas as pd

df = pd.read_csv('../multi_sensor_iot_dataset2.csv')

print('=' * 60)
print('DATASET ANALYSIS: multi_sensor_iot_dataset2.csv')
print('=' * 60)

print(f'\nDataset Shape: {df.shape}')
print(f'Rows: {df.shape[0]}, Columns: {df.shape[1]}')

print('\nColumn Names:')
print(df.columns.tolist())

print('\nData Types:')
print(df.dtypes)

print('\n' + '='*60)
print('ATTACK DISTRIBUTION')
print('='*60)
attack_counts = df['attack_type'].value_counts().sort_index()
print(attack_counts)
print('\nAttack Type Percentages:')
for attack_type, count in attack_counts.items():
    pct = (count / len(df)) * 100
    print(f"  {attack_type:20s}: {count:5d} ({pct:5.2f}%)")

print('\n' + '='*60)
print('SENSOR DISTRIBUTION')
print('='*60)
sensor_counts = df['sensor_id'].value_counts().sort_index()
print(sensor_counts)

print('\n' + '='*60)
print('SENSOR × ATTACK CROSS-TABULATION')
print('='*60)
crosstab = pd.crosstab(df['sensor_id'], df['attack_type'])
print(crosstab)

print('\n' + '='*60)
print('MEASUREMENT RANGES')
print('='*60)
print(f"Temperature Range: {df['temperature_c'].min():.2f}°C to {df['temperature_c'].max():.2f}°C")
print(f"Humidity Range:    {df['humidity_percent'].min():.2f}% to {df['humidity_percent'].max():.2f}%")

print('\nTemperature Stats:')
print(f"  Mean: {df['temperature_c'].mean():.2f}°C")
print(f"  Std:  {df['temperature_c'].std():.2f}°C")

print('\nHumidity Stats:')
print(f"  Mean: {df['humidity_percent'].mean():.2f}%")
print(f"  Std:  {df['humidity_percent'].std():.2f}%")

print('\n' + '='*60)
print('TIMESTAMP INFO')
print('='*60)
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"Start: {df['timestamp'].min()}")
print(f"End:   {df['timestamp'].max()}")
print(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).total_seconds():.0f} seconds")

print('\n' + '='*60)
print('FIRST 10 ROWS')
print('='*60)
print(df.head(10).to_string())

print('\n✅ Analysis Complete')

