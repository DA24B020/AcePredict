"""ML pipeline console: DAG, stage status and throughput, runs, errors, admin actions."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import api_client as api
import ui

t0 = ui.page("Pipeline console", "🛠️", "pipeline")
st.title("🛠️ ML pipeline console")
st.caption("Data ingestion → validation → Spark transform → Ray Tune → Elo → features → "
           "training → evaluation → promotion gate → serving snapshot. Orchestrated by DVC.")

links = st.columns(4)
links[0].link_button("MLflow (experiments & registry)", ui.MLFLOW_URL)
links[1].link_button("Grafana (dashboards)", ui.GRAFANA_URL)
links[2].link_button("Prometheus (metrics & alerts)", ui.PROMETHEUS_URL)
links[3].link_button("Ray dashboard (during tuning)", ui.RAY_URL)

if st.button("🔄 Refresh"):
    st.rerun()
s, err = api.get("/pipeline/status")
if err:
    st.error(err)
    st.stop()

state = "running" if s["running"] else ("queued" if s["queued"] else "idle")
st.info(f"Pipeline is **{state}**.")
st.subheader("Pipeline DAG (from dvc.yaml, coloured by last status)")
st.graphviz_chart(s["dag_dot"], use_container_width=True)

st.subheader("Stages: status, speed and throughput")
stages = pd.DataFrame(s["stages"])
st.dataframe(stages[["name", "status", "finished_at", "duration_s", "rows_in", "rows_out",
                     "rows_per_s", "error"]], hide_index=True, use_container_width=True)
if s.get("tuning"):
    t = s["tuning"]
    st.caption(f"Ray Tune: {t['trials']} trials in {t['seconds']} s "
               f"({t['trials_per_minute']} trials/min), best validation log-loss "
               f"{t['best_val_log_loss']:.4f}.")

st.subheader("Pipeline runs (successes and failures)")
runs = pd.DataFrame(s["runs"])
st.dataframe(runs, hide_index=True, use_container_width=True) if not runs.empty else st.write(
    "No runs triggered from the console yet.")
if s["failed_log_tail"]:
    with st.expander("Log of the last failed run", expanded=True):
        st.code("\n".join(s["failed_log_tail"]))

st.subheader("Recent MLflow runs")
mr = pd.DataFrame(s["mlflow_runs"])
if not mr.empty:
    mr["metrics"] = mr["metrics"].apply(lambda m: ", ".join(f"{k}={v}" for k, v in m.items()))
    st.dataframe(mr, hide_index=True, use_container_width=True)

if s.get("metrics"):
    m = s["metrics"]
    c = st.columns(4)
    c[0].metric("Test accuracy", f"{m['test']['accuracy']:.3f}")
    c[1].metric("Test log-loss", f"{m['test']['log_loss']:.4f}")
    c[2].metric("Elo-only log-loss", f"{m['baselines']['elo_only']['log_loss']:.4f}")
    c[3].metric("Beats baselines", "Yes" if m["beats_baselines"] else "No")

info, _ = api.get("/model/info")
st.subheader("Admin actions")
st.write(f"Champion: **v{(info or {}).get('champion_version')}** · previous: "
         f"v{(info or {}).get('previous_version')} · challenger: "
         f"v{(info or {}).get('challenger_version')}")
key = st.text_input("Admin API key", type="password")
a1, a2, a3 = st.columns(3)
if a1.button("▶ Run pipeline", disabled=not key):
    r, err = api.post("/admin/pipeline/run", {"reason": "console"}, api_key=key)
    st.error(err) if err else st.success(f"Queued run {r['run_id']}")
version = a2.text_input("Version to promote", max_chars=10)
if a2.button("⬆ Promote", disabled=not (key and version)):
    r, err = api.post("/admin/model/promote", {"version": version}, api_key=key)
    st.error(err) if err else st.success(f"Champion is now v{r['champion_version']}")
if a3.button("↩ Roll back", disabled=not key):
    r, err = api.post("/admin/model/rollback", api_key=key)
    st.error(err) if err else st.success(f"Champion is now v{r['champion_version']}")
ui.done("pipeline", t0)
