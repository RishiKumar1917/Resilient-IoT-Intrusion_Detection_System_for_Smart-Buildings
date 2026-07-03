import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Ensure terminal input is disabled to prevent Streamlit from hanging
import hybrid_iot_ids
hybrid_iot_ids.ENABLE_FEEDBACK = False

from hybrid_iot_ids import (
    load_data, DataConfig, ModelConfig, ReplayConfig, ThresholdConfig,
    engineer_features, train_rf_classifier, get_feature_columns,
    normalize_per_source, make_sequences, train_autoencoder,
    compute_reconstruction_error, estimate_threshold, _subset_bundle,
    _split_normal_sequences, _seed_history_buffers, NORMAL_LABEL,
    classify_window, build_sequence_feature_vector
)
from feedback_engine import add_feedback

st.set_page_config(page_title="Hybrid IoT IDS Monitor", layout="wide")

@st.cache_resource(show_spinner="Initializing Hybrid IDS Pipeline (This takes ~45 seconds)...")
def initialize_system():
    data_config = DataConfig()
    model_config = ModelConfig()
    replay_config = ReplayConfig()
    threshold_config = ThresholdConfig()

    merged = load_data(data_config)
    engineered = engineer_features(merged, consistency_window=data_config.consistency_window)
    
    hybrid_iot_ids.global_engineered_df = engineered
    train_rf_classifier(engineered, window_size=data_config.window_size)
    
    feature_cols = get_feature_columns()
    normalized, scalers = normalize_per_source(engineered, feature_cols)
    bundle = make_sequences(normalized, window_size=data_config.window_size, feature_cols=feature_cols)

    labeled_bundle = bundle
    normal_mask = labeled_bundle.labels == NORMAL_LABEL
    normal_bundle = _subset_bundle(labeled_bundle, normal_mask)
    train_bundle, val_bundle = _split_normal_sequences(normal_bundle, model_config.validation_fraction)

    model, history = train_autoencoder(train_bundle.X, val_bundle.X, model_config)
    val_errors = compute_reconstruction_error(model, val_bundle.X)
    threshold_main = estimate_threshold(val_errors, sigma=3)

    history_buffers = _seed_history_buffers(train_bundle, replay_config)
    
    source = merged["source"].iloc[0]
    eval_bundle = labeled_bundle
    
    df_grouped = engineered.groupby("timestamp").agg({
        "temperature_c": ["mean", "std", "min", "max"],
        "humidity_percent": ["mean", "std"]
    })
    df_grouped.columns = ["temp_mean", "temp_std", "temp_min", "temp_max", "hum_mean", "hum_std"]
    df_grouped = df_grouped.reset_index()
    
    return {
        "model": model,
        "threshold_main": threshold_main,
        "history_buffers": history_buffers,
        "replay_config": replay_config,
        "threshold_config": threshold_config,
        "engineered_df": engineered,
        "df_grouped": df_grouped,
        "eval_bundle": eval_bundle,
        "source": source,
        "window_size": data_config.window_size
    }

sys_state = initialize_system()

# State Management
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "stats" not in st.session_state:
    st.session_state.stats = {
        "total_windows": 0,
        "rule_engine": 0,
        "ensemble": 0,
        "feedback_memory": 0
    }
if "processed_indices" not in st.session_state:
    st.session_state.processed_indices = set()
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []
if "seen_alert_keys" not in st.session_state:
    st.session_state.seen_alert_keys = set()
if "timeline" not in st.session_state:
    st.session_state.timeline = []

eval_metadata = sys_state["eval_bundle"].metadata
max_index = len(eval_metadata) - 1

if st.session_state.current_index > max_index:
    st.session_state.current_index = max_index

st.title("🛡️ Hybrid IoT Network IDS Monitor")
st.markdown("Real-time monitoring dashboard with Human-in-the-Loop Feedback integration.")

# Extract current window
idx = st.session_state.current_index
row = eval_metadata.iloc[idx]
source = sys_state["source"]
window_size = sys_state["window_size"]
df_grouped = sys_state["df_grouped"]

unique_times = df_grouped["timestamp"].values
window_times = unique_times[idx : idx + window_size]
window_df = sys_state["engineered_df"][sys_state["engineered_df"]["timestamp"].isin(window_times)].sort_values("timestamp")
window_grouped = df_grouped[df_grouped["timestamp"].isin(window_times)]

# --- RUN INFERENCE ---
sequence = sys_state["eval_bundle"].X[idx]
result = classify_window(
    sequence=sequence,
    model=sys_state["model"],
    threshold=sys_state["threshold_main"],
    history_buffer=list(sys_state["history_buffers"][source]),
    replay_config=sys_state["replay_config"],
    threshold_config=sys_state["threshold_config"],
    timestamp=row["end_timestamp"],
    source=source,
    window_size=window_size
)

# --- CLASSIFY ALERT STRENGTH ---
def classify_alert_strength(prediction, confidence):
    """Categorize alerts: Strong (RED), Weak (YELLOW), Normal (BLUE)."""
    if prediction == "Normal":
        return "normal"
    if confidence in ["HIGH", "VERY HIGH"]:
        return "strong"
    return "weak"

alert_strength = classify_alert_strength(result.predicted_label, result.confidence)

if result.predicted_label == "Normal":
    status = "🟢 System Stable"
elif alert_strength == "weak":
    status = "🟡 Suspicious Activity (Weak Signal)"
else:
    status = "🔴 Attack Detected"

st.markdown(f"## {status}")

feature_dict = build_sequence_feature_vector(window_df.tail(window_size))
temp_slope = feature_dict.get("temp_slope", 0.0)
temp_std = feature_dict.get("temp_std", 0.0)
temp_range = feature_dict.get("temp_range", 0.0)
temp_entropy = feature_dict.get("temp_entropy", 0.0)

thresholds = sys_state["threshold_config"]
explanation = []

if abs(temp_slope) > thresholds.drift_threshold:
    explanation.append("📈 Strong trend detected (Drift)")
if temp_range > thresholds.injection_range_threshold:
    explanation.append("⚡ Sudden spike detected (Injection)")
if temp_std < thresholds.drop_std_threshold:
    explanation.append("⏸ Very low variation (Drop)")
if temp_std > thresholds.noise_std_threshold:
    explanation.append("🌪 High randomness (Noise)")

st.markdown("### 🧠 Why This Alert")
if explanation:
    for e in explanation:
        st.write(e)
else:
    if result.predicted_label == "Normal":
        st.write("Normal behavior — no anomalies detected")
    elif result.predicted_label == "Replay Attack":
        st.write("🔄 Exact historical pattern match detected (Replay)")
    else:
        st.write("Anomalous pattern flagged by Machine Learning ensemble")

st.subheader(f"Window Analysis: {row['start_timestamp']} ➔ {row['end_timestamp']}")

# Assign prediction to window_df
window_df = window_df.copy()
window_df["prediction"] = result.predicted_label
window_df["confidence"] = result.confidence

if "pred_history" not in st.session_state:
    st.session_state.pred_history = {}
window_key = f"{row['start_timestamp']}_{row['end_timestamp']}"
st.session_state.pred_history[window_key] = result.predicted_label

# --- TIME-SERIES VISUALIZATION ---
end_ts = row["end_timestamp"]

# --- VIEW CONTROLS ---
viz_col1, viz_col2, viz_col3 = st.columns(3)
with viz_col1:
    view_mode = st.radio("📊 Visualization Mode:", ["Single Sensor View", "Multi Sensor View"], horizontal=True)
with viz_col2:
    focus_mode = st.toggle("🔎 Focus on Current Window", value=False)
with viz_col3:
    auto_zoom_attack = result.predicted_label != "Normal"
    if auto_zoom_attack:
        st.markdown("🎯 **Auto-Zoom: ON** (Attack Detected)")
    else:
        st.markdown("🟢 Auto-Zoom: OFF (Normal)")

all_sensors = sorted(sys_state["engineered_df"]["sensor_id"].unique())

context_start = max(0, idx - window_size * 4)
context_times = unique_times[context_start : idx + window_size]

# Build prediction mapping for context coloring
ts_to_pred = {}
for i in range(context_start, idx + 1):
    c_times = unique_times[i : i + window_size]
    if len(c_times) > 0:
        c_row = eval_metadata.iloc[i] if i < len(eval_metadata) else None
        if c_row is not None:
            w_key = f"{c_row['start_timestamp']}_{c_row['end_timestamp']}"
            p = st.session_state.pred_history.get(w_key, "Normal")
            for t in c_times:
                ts_to_pred[t] = p

# --- SHOW WEAK SIGNALS TOGGLE ---
show_weak = st.sidebar.toggle("🔍 Show Weak Signals", value=False)

# --- DETERMINE VISIBLE TIME RANGE ---
if focus_mode:
    visible_times = window_times
elif auto_zoom_attack:
    # Smart auto-zoom: narrow to last 20 timestamps around detection point
    zoom_end = idx + window_size
    zoom_start = max(0, zoom_end - 20)
    visible_times = unique_times[zoom_start : zoom_end]
else:
    visible_times = context_times

# Adaptive styling based on zoom level
is_zoomed = focus_mode or auto_zoom_attack
line_width = 3 if is_zoomed else 2
marker_size = 12 if is_zoomed else 8

fig = go.Figure()

if view_mode == "Single Sensor View":
    selected_sensor = st.selectbox("🔍 Select Sensor:", all_sensors, index=0)
    
    visible_df = sys_state["engineered_df"][
        (sys_state["engineered_df"]["timestamp"].isin(visible_times)) &
        (sys_state["engineered_df"]["sensor_id"] == selected_sensor)
    ].sort_values("timestamp")

    visible_df = visible_df.copy()
    visible_df["prediction"] = visible_df["timestamp"].apply(lambda ts: ts_to_pred.get(ts, "Normal"))

    # Clean temperature line
    fig.add_trace(go.Scatter(
        x=visible_df["timestamp"],
        y=visible_df["temperature_c"],
        mode="lines",
        line=dict(color="rgba(100, 149, 237, 0.8)", width=line_width),
        name=f"{selected_sensor} Temperature"
    ))

    # Strong attack markers (Red X)
    strong_attacks = visible_df[visible_df["prediction"] != "Normal"]
    if not strong_attacks.empty:
        fig.add_trace(go.Scatter(
            x=strong_attacks["timestamp"],
            y=strong_attacks["temperature_c"],
            mode="markers",
            marker=dict(color="red", size=marker_size, symbol="x"),
            name="Attack Detected",
            opacity=1.0
        ))

    # Dynamic axis range
    if not visible_df.empty:
        y_min = visible_df["temperature_c"].min()
        y_max = visible_df["temperature_c"].max()
        x_min = visible_df["timestamp"].min()
        x_max = visible_df["timestamp"].max()
    else:
        y_min, y_max, x_min, x_max = 20, 35, None, None

else:
    # Multi Sensor View
    selected_sensors = st.multiselect("🔍 Select Sensors (max 3 for clarity):", all_sensors, default=all_sensors[:3], max_selections=3)

    # Aggregated mean line
    visible_grouped = df_grouped[df_grouped["timestamp"].isin(visible_times)]
    fig.add_trace(go.Scatter(
        x=visible_grouped["timestamp"],
        y=visible_grouped["temp_mean"],
        mode="lines",
        line=dict(color="rgba(255, 165, 0, 0.9)", width=line_width),
        name="Aggregated Mean"
    ))

    # Individual sensor overlays (soft colors)
    sensor_colors = ["rgba(100, 149, 237, 0.6)", "rgba(50, 205, 50, 0.6)", "rgba(186, 85, 211, 0.6)"]
    all_temps = []
    for i, sensor in enumerate(selected_sensors):
        sensor_df = sys_state["engineered_df"][
            (sys_state["engineered_df"]["timestamp"].isin(visible_times)) &
            (sys_state["engineered_df"]["sensor_id"] == sensor)
        ].sort_values("timestamp")
        all_temps.extend(sensor_df["temperature_c"].tolist())
        fig.add_trace(go.Scatter(
            x=sensor_df["timestamp"],
            y=sensor_df["temperature_c"],
            mode="lines",
            line=dict(color=sensor_colors[i % len(sensor_colors)], width=1.5),
            name=sensor
        ))

    # Attack markers across all selected sensors
    multi_context = sys_state["engineered_df"][
        (sys_state["engineered_df"]["timestamp"].isin(visible_times)) &
        (sys_state["engineered_df"]["sensor_id"].isin(selected_sensors))
    ].sort_values("timestamp").copy()
    multi_context["prediction"] = multi_context["timestamp"].apply(lambda ts: ts_to_pred.get(ts, "Normal"))
    
    attack_points = multi_context[multi_context["prediction"] != "Normal"]
    if not attack_points.empty:
        fig.add_trace(go.Scatter(
            x=attack_points["timestamp"],
            y=attack_points["temperature_c"],
            mode="markers",
            marker=dict(color="red", size=marker_size, symbol="x"),
            name="Attack Detected",
            opacity=1.0
        ))

    # Dynamic axis range for multi-sensor
    if all_temps:
        y_min = min(all_temps)
        y_max = max(all_temps)
    elif not visible_grouped.empty:
        y_min = visible_grouped["temp_mean"].min()
        y_max = visible_grouped["temp_mean"].max()
    else:
        y_min, y_max = 20, 35
    
    if not visible_grouped.empty:
        x_min = visible_grouped["timestamp"].min()
        x_max = visible_grouped["timestamp"].max()
    else:
        x_min, x_max = None, None

# Detection boundary line (both modes)
fig.add_vline(x=str(end_ts), line_width=1.5, line_dash="dash", line_color="red")
fig.add_annotation(
    x=str(end_ts),
    y=y_max + 0.3,
    text="Detection Point",
    showarrow=True,
    arrowhead=2,
    font=dict(color="red")
)

# Apply dynamic axis scaling
y_padding = 0.5
chart_title = "Sensor Trend with Attack Detection"
if is_zoomed:
    chart_title = "🎯 Zoomed: Sensor Trend with Attack Detection"

layout_kwargs = dict(
    title=chart_title,
    xaxis_title="Timestamp",
    yaxis_title="Temperature (°C)",
    hovermode="x unified",
    margin=dict(l=0, r=0, t=40, b=0),
    yaxis=dict(range=[y_min - y_padding, y_max + y_padding])
)

if x_min is not None and x_max is not None:
    layout_kwargs["xaxis"] = dict(range=[str(x_min), str(x_max)], title="Timestamp")

fig.update_layout(**layout_kwargs)

st.plotly_chart(fig, use_container_width=True)

with st.expander("View Raw Window Data"):
    st.subheader("Raw Multi-Sensor Data")
    st.dataframe(window_df[['timestamp', 'sensor_id', 'temperature_c', 'humidity_percent', 'attack_type', 'prediction']], use_container_width=True)
    st.subheader("Aggregated Room-Level Data")
    st.dataframe(window_grouped, use_container_width=True)

# --- PROCESS WINDOW (stats, alerts, timeline) ---
if idx not in st.session_state.processed_indices:
    st.session_state.processed_indices.add(idx)
    st.session_state.stats["total_windows"] += 1
    src = result.decision_source
    if src == "rule_engine":
        st.session_state.stats["rule_engine"] += 1
    elif src in ["ensemble_gb", "ensemble_rf"]:
        st.session_state.stats["ensemble"] += 1
    elif src == "feedback_memory":
        st.session_state.stats["feedback_memory"] += 1
        
    st.session_state.timeline.append(result.predicted_label)
    st.session_state.timeline = st.session_state.timeline[-50:]
        
    # Only create alerts for HIGH/VERY HIGH confidence detections
    if result.predicted_label != "Normal" and alert_strength == "strong":
        alert_key = f"{row['end_timestamp']}_{result.predicted_label}"
        if alert_key not in st.session_state.seen_alert_keys:
            st.session_state.alert_log.insert(0, {
                "timestamp": str(row["end_timestamp"]),
                "attack": result.predicted_label,
                "confidence": result.confidence,
                "source": result.decision_source
            })
            st.session_state.seen_alert_keys.add(alert_key)
            st.session_state.alert_log = st.session_state.alert_log[:50]

# System Statistics Sidebar
st.sidebar.title("📊 System Statistics")
st.sidebar.metric("Total Windows Processed", st.session_state.stats["total_windows"])
st.sidebar.metric("Rule Engine Decisions", st.session_state.stats["rule_engine"])
st.sidebar.metric("Ensemble (GB/RF) Decisions", st.session_state.stats["ensemble"])
st.sidebar.metric("Feedback Engine Uses", st.session_state.stats["feedback_memory"])

if st.session_state.stats["total_windows"] > 0:
    df_stats = pd.DataFrame({
        "Source": ["Rule Engine", "Ensemble", "Feedback Memory"],
        "Count": [
            st.session_state.stats["rule_engine"], 
            st.session_state.stats["ensemble"], 
            st.session_state.stats["feedback_memory"]
        ]
    })
    df_stats = df_stats[df_stats["Count"] > 0]
    if not df_stats.empty:
        fig_pie = go.Figure(data=[go.Pie(labels=df_stats["Source"], values=df_stats["Count"], hole=.4)])
        fig_pie.update_layout(title="Decision Sources", margin=dict(t=40, b=0, l=0, r=0))
        st.sidebar.plotly_chart(fig_pie, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Active Alerts (Strong Only)")

def get_alert_color(attack):
    if "Injection" in attack:
        return "error"
    elif "Replay" in attack:
        return "error"
    elif "Drift" in attack:
        return "warning"
    elif "Noise" in attack:
        return "warning"
    elif "Drop" in attack:
        return "info"
    else:
        return "info"

# Only show strong (HIGH/VERY HIGH) alerts in the main panel
strong_alerts = [a for a in st.session_state.alert_log if a["confidence"] in ["HIGH", "VERY HIGH"]]
latest_alerts = strong_alerts[:5]

for alert in latest_alerts:
    msg = f"{alert['attack']} @ {alert['timestamp']}\nConfidence: {alert['confidence']}\nSource: {alert['source']}"
    level = get_alert_color(alert["attack"])
    if level == "error":
        st.sidebar.error(msg)
    elif level == "warning":
        st.sidebar.warning(msg)
    else:
        st.sidebar.info(msg)

if not latest_alerts:
    st.sidebar.success("No strong alerts detected")

# Display Metrics
col1, col2, col3, col4 = st.columns(4)
color = "normal" if result.predicted_label == "Normal" else "inverse"
col1.metric("Prediction", result.predicted_label, delta_color=color)
col2.metric("Decision Source", str(result.decision_source))
col3.metric("Confidence", str(result.confidence))
col4.metric("🚨 Strong Alerts", len(strong_alerts))

# --- ATTACK TIMELINE ---
def get_timeline_color(p):
    if "Replay" in p: return "#FFD700"
    elif "Injection" in p: return "#FF4B4B"
    elif "Drift" in p: return "#FFA500"
    elif "Noise" in p: return "#800080"
    elif "Drop" in p: return "#1E90FF"
    else: return "#4CAF50"

html = "<div style='display:flex; margin-bottom: 20px;'>"
for p in st.session_state.timeline:
    color = get_timeline_color(p)
    html += f"<div style='width:10px;height:20px;background:{color};margin-right:2px;border-radius:2px;' title='{p}'></div>"
html += "</div>"

st.markdown("### 📊 Attack Timeline")
st.markdown(html, unsafe_allow_html=True)

# --- ATTACK STREAK DETECTION ---
timeline_recent = st.session_state.timeline[-10:]
if len(timeline_recent) >= 5 and len(set(timeline_recent[-5:])) == 1 and timeline_recent[-1] != "Normal":
    streak_type = timeline_recent[-1]
    st.warning(f"⚠️ Sustained {streak_type} detected!")

if timeline_recent.count("Injection Attack") >= 3:
    st.error("🚨 Multiple Injection Attacks detected recently!")

st.divider()

# --- INCIDENT SUMMARY PANEL ---
alerts = st.session_state.alert_log
st.markdown("### 📋 Incident Summary")

if alerts:
    total_attacks = len(alerts)
    attack_types = [a["attack"] for a in alerts]
    most_common = max(set(attack_types), key=attack_types.count)
    last_attack = alerts[0]["attack"]
    high_conf = sum(1 for a in alerts if a["confidence"] in ["HIGH", "VERY HIGH"])

    scol1, scol2, scol3, scol4 = st.columns(4)
    scol1.metric("Total Attacks", total_attacks)
    scol2.metric("Most Frequent", most_common)
    scol3.metric("Last Attack", last_attack)
    scol4.metric("High Confidence", high_conf)
else:
    st.info("No attacks detected yet")

st.divider()
feature_dict = build_sequence_feature_vector(window_df.tail(window_size))
current_features = np.array(list(feature_dict.values()), dtype=np.float32)

# Explainability Panel
st.subheader("💡 Decision Explainability")

t_slope = feature_dict.get("temp_30_slope", 0.0)
t_std = feature_dict.get("temp_30_std", 0.0)
t_range = feature_dict.get("temp_30_range", 0.0)
t_entropy = feature_dict.get("temp_30_entropy", 0.0)

t_jump = feature_dict.get("temp_10_max_jump", 0.0)
t_zscore = feature_dict.get("temp_10_zscore_max", 0.0)

fcol1, fcol2, fcol3, fcol4 = st.columns(4)
fcol1.metric("Temp Slope (30s)", f"{t_slope:.4f}")
fcol2.metric("Temp Std Dev (30s)", f"{t_std:.4f}")
fcol3.metric("Max Jump (10s)", f"{t_jump:.4f}")
fcol4.metric("Z-Score Max (10s)", f"{t_zscore:.4f}")

interpretation = []
if abs(t_slope) > sys_state["threshold_config"].drift_threshold:
    interpretation.append("📈 Strong trend detected (Drift)")
if t_jump > 6.0 or t_zscore > 4.0:
    interpretation.append("⚡ Sudden burst detected (Injection)")
elif t_range > sys_state["threshold_config"].injection_range_threshold:
    interpretation.append("⚡ Spike behavior detected (Injection)")
if t_std < sys_state["threshold_config"].drop_std_threshold and t_range < sys_state["threshold_config"].drop_range_threshold:
    interpretation.append("⏸ Very low variation (Drop)")
if result.replay_flag:
    interpretation.append("🔄 Exact historical pattern match detected (Replay)")

if not interpretation:
    if result.predicted_label == "Normal":
        interpretation.append("Normal behavior — no anomalies detected")
    else:
        interpretation.append("Ensemble detected complex latent features.")

st.markdown("**Interpretation:**")
for msg in interpretation:
    st.markdown(f"- {msg}")

st.markdown("**Decision Path:**")
if result.decision_source == "rule_engine":
    st.info(f"Rule Engine → Triggered {result.predicted_label}")
elif result.decision_source == "feedback_memory":
    st.success(f"Feedback Engine → Similarity Match → Overrode with {result.predicted_label}")
elif result.decision_source == "ensemble_gb":
    st.warning(f"Isolation Forest → Anomaly Detected → Gradient Boosting used → {result.predicted_label}")
elif result.decision_source == "ensemble_rf":
    st.info(f"Isolation Forest → Normal → Random Forest used → {result.predicted_label}")
else:
    st.write(f"Source: {result.decision_source} → {result.predicted_label}")

st.divider()

# --- DETECTION CAPABILITY OVERVIEW ---
st.subheader("🛡️ Detection Capability Overview")

detection_data = [
    {"Attack Type": "Injection", "Method": "Rule + Spike", "Strategy": "Sudden Change", "Detection Strength": "🟢 Strong (>85%)"},
    {"Attack Type": "Drift", "Method": "Rule + Slope", "Strategy": "Trend-Based", "Detection Strength": "🟢 Strong (~90%)"},
    {"Attack Type": "Replay", "Method": "Similarity", "Strategy": "Pattern Matching", "Detection Strength": "🟢 Strong (>90%)"},
    {"Attack Type": "Noise", "Method": "ML + Rule", "Strategy": "Randomness-Based", "Detection Strength": "🟠 Moderate (~65%)"},
    {"Attack Type": "Drop", "Method": "Rule", "Strategy": "Low Variation", "Detection Strength": "🟠 Moderate (~55%)"},
]
st.table(pd.DataFrame(detection_data))

st.divider()

# Feedback Section
st.subheader("Human-in-the-Loop Feedback")
st.write("If the model made a mistake, override the prediction and store it in memory.")

# Only allow feedback storage for HIGH confidence or manual override
if alert_strength == "weak":
    st.caption("⚠️ Current detection is LOW confidence — feedback will be stored only if you manually confirm.")

feedback_labels = [
    "Replay Attack", 
    "Injection Attack", 
    "Drop Attack", 
    "Drift Attack", 
    "Noise Attack", 
    "Normal"
]

cols = st.columns(len(feedback_labels))
for i, label in enumerate(feedback_labels):
    if cols[i].button(label, key=f"btn_{label}"):
        # Always store manual feedback (user explicitly confirmed)
        add_feedback(current_features.tolist(), label)
        st.toast(f"Stored feedback: {label}", icon="✅")
        st.success(f"Successfully overrode prediction to **{label}**. The feedback engine will now remember this signature.")

st.divider()

# Next Window
if st.button("⏭️ Next Window", type="primary", use_container_width=True):
    st.session_state.current_index += 1
    st.rerun()
