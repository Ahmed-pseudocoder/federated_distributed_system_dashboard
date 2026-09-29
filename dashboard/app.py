from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"


st.set_page_config(page_title="Federated Learning Control Plane", page_icon="◉", layout="wide")
st.title("Federated Learning Control Plane")
st.caption("Live state is read from SQLite events emitted by the coordinator and client processes.")


def query(database: Path, sql: str, params: tuple = ()) -> pd.DataFrame:
    if not database.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=2) as connection:
            return pd.read_sql_query(sql, connection, params=params)
    except (sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame()


def load_json(value, fallback=None):
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback if fallback is not None else {}


databases = sorted(
    [path for path in RESULTS.glob("*.db") if path.name != "hpo.db"],
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
if not databases:
    st.info("No run database exists yet. Start one with `python run_fl.py`.")
    st.stop()

selected_name = st.sidebar.selectbox("Run database", [path.name for path in databases])
database = RESULTS / selected_name
if st.sidebar.button("Refresh now", width="stretch"):
    st.rerun()
st.sidebar.caption(f"Reading {database}")

state_frame = query(database, "SELECT value_json,updated_at FROM state WHERE key='current'")
state = load_json(state_frame.iloc[0]["value_json"]) if not state_frame.empty else {}
workers_frame = query(database, "SELECT client_id,updated_at,payload_json FROM workers ORDER BY client_id")
workers = []
for _, row in workers_frame.iterrows():
    payload = load_json(row["payload_json"])
    payload["client_id"] = int(row["client_id"])
    workers.append(payload)

st.subheader("Global training")
metric_columns = st.columns(6)
current_round = state.get("round", 0)
total_rounds = state.get("total_rounds", "—")
metric_columns[0].metric("Round", f"{current_round} / {total_rounds}")
metric_columns[1].metric("Accuracy", f"{100 * state.get('final_accuracy', state.get('accuracy', 0)):.2f}%")
metric_columns[2].metric("Loss", f"{state.get('final_loss', state.get('loss', 0)):.4f}")
metric_columns[3].metric("Active clients", sum(1 for worker in workers if worker.get("available")))
metric_columns[4].metric("Failed clients", sum(1 for worker in workers if worker.get("status") == "FAILED"))
metric_columns[5].metric("Status", state.get("status", "UNKNOWN"))
if state.get("duration") is not None:
    st.caption(f"Latest round duration: {state['duration']:.2f}s")

st.subheader("Live worker monitor")
if workers:
    columns = st.columns(len(workers))
    now = time.time()
    for column, worker in zip(columns, workers):
        status = worker.get("status", "UNKNOWN")
        icon = "🟢" if status in {"HEALTHY", "TRAINING"} else "🔴" if status == "FAILED" else "🟡"
        heartbeat_age = max(0.0, now - float(worker.get("last_heartbeat", now)))
        with column:
            st.markdown(f"### {icon} Client {worker['client_id']}")
            st.write(f"**{status}** · PID {worker.get('pid', '—')}")
            st.write(
                f"Round {worker.get('current_round', 0)} · Epoch {worker.get('current_epoch', 0)} · Batch {worker.get('current_batch', 0)}"
            )
            st.write(f"Step {worker.get('global_step', 0)} · heartbeat {heartbeat_age:.1f}s ago")
            loss = worker.get("last_loss")
            accuracy = worker.get("last_accuracy")
            st.write(f"Loss {loss:.4f}" if loss is not None else "Loss —")
            st.write(f"Accuracy {100 * accuracy:.2f}%" if accuracy is not None else "Accuracy —")
            st.caption(f"Reliability {worker.get('reliability', 0):.2f}")
else:
    st.info("Workers have not registered yet.")

st.subheader("Failure investigation")
failure_frame = query(
    database,
    "SELECT timestamp,client_id,round,payload_json FROM events WHERE event_type='WORKER_FAILURE' ORDER BY id DESC",
)
if failure_frame.empty:
    st.success("No worker failure has been detected in this run.")
else:
    for _, row in failure_frame.iterrows():
        failure = load_json(row["payload_json"])
        with st.expander(
            f"WORKER FAILURE DETECTED — Client {int(row['client_id'])}, Round {int(row['round'])}",
            expanded=True,
        ):
            fields = {
                "Client ID": int(row["client_id"]),
                "Failure Type": failure.get("failure_type"),
                "Round": int(row["round"]),
                "Epoch": failure.get("epoch"),
                "Batch": failure.get("batch"),
                "Global Step": failure.get("global_step"),
                "Last Loss": failure.get("last_loss"),
                "Last Accuracy": failure.get("last_accuracy"),
                "Last Heartbeat": failure.get("last_heartbeat"),
                "Detected At": failure.get("detected_at"),
                "Detection Latency": f"{failure.get('detection_latency_ms', 0):.1f} ms",
                "Checkpoint": failure.get("checkpoint_used"),
                "Recovery Status": failure.get("status"),
                "Recovery Time": f"{failure.get('recovery_time_ms', 0):.1f} ms",
                "Incomplete Update Discarded": failure.get("incomplete_update_discarded"),
                "Process Exit Code": failure.get("process_exitcode"),
            }
            left, right = st.columns(2)
            for index, (label, value) in enumerate(fields.items()):
                (left if index % 2 == 0 else right).write(f"**{label}:** {value}")

st.subheader("Event timeline")
timeline = query(
    database,
    "SELECT timestamp,event_type,source,client_id,round,payload_json FROM events ORDER BY id DESC LIMIT 200",
)
if not timeline.empty:
    timeline = timeline.iloc[::-1]
    for _, event in timeline.iterrows():
        stamp = str(event["timestamp"])[11:23]
        subject = f"Client-{int(event['client_id'])}" if pd.notna(event["client_id"]) else "Coordinator"
        round_text = f" · Round {int(event['round'])}" if pd.notna(event["round"]) else ""
        st.markdown(f"`{stamp}` **{subject}** — {event['event_type'].replace('_', ' ').title()}{round_text}")
else:
    st.info("No events recorded yet.")

st.subheader("Training charts")
rounds = query(database, "SELECT * FROM rounds ORDER BY round")
telemetry = query(database, "SELECT * FROM telemetry ORDER BY id")
chart_left, chart_right = st.columns(2)
if not rounds.empty:
    with chart_left:
        st.plotly_chart(px.line(rounds, x="round", y="accuracy", markers=True, title="Global accuracy by round"), width="stretch")
        st.plotly_chart(px.bar(rounds, x="round", y="active_clients", title="Healthy clients by round"), width="stretch")
    with chart_right:
        st.plotly_chart(px.line(rounds, x="round", y="loss", markers=True, title="Global loss by round"), width="stretch")
        st.plotly_chart(px.line(rounds, x="round", y="duration", markers=True, title="Round duration"), width="stretch")
if not telemetry.empty:
    telemetry["client"] = telemetry["client_id"].map(lambda value: f"Client {value}")
    st.plotly_chart(
        px.line(telemetry, x="global_step", y="loss", color="client", title="Client local training loss"),
        width="stretch",
    )
if rounds.empty and telemetry.empty:
    st.info("Charts will appear after the first round and telemetry samples are persisted.")

st.subheader("HPO results")
hpo_database = RESULTS / "hpo.db"
hpo = query(hpo_database, "SELECT * FROM hpo_trials ORDER BY trial_id")
if hpo.empty:
    st.info("No HPO results yet. Run `python run_hpo.py`.")
else:
    completed = hpo[hpo["status"] == "COMPLETED"]
    best_id = int(completed.loc[completed["accuracy"].idxmax(), "trial_id"]) if not completed.empty else None
    display = hpo.rename(
        columns={
            "trial_id": "Trial", "learning_rate": "Learning Rate", "batch_size": "Batch Size",
            "local_epochs": "Local Epochs", "accuracy": "Accuracy", "duration": "Time", "status": "Status",
        }
    )[["Trial", "Learning Rate", "Batch Size", "Local Epochs", "Accuracy", "Time", "Status"]]
    st.dataframe(
        display.style.apply(
            lambda row: ["background-color: #d8f3dc" if best_id == row["Trial"] else "" for _ in row], axis=1
        ),
        width="stretch",
        hide_index=True,
    )
    if best_id is not None:
        st.success(f"Best actual trial: {best_id}")

