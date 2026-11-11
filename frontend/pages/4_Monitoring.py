"""Live monitoring: accuracy on real results, feature drift, and the Grafana dashboards."""

from __future__ import annotations

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import api_client as api
import ui

t0 = ui.page("Monitoring", "📈", "monitoring")
st.title("📈 Monitoring")
s, err = api.get("/monitoring/summary")
if err:
    st.error(err)
else:
    c = st.columns(4)
    c[0].metric("Predictions logged", s["predictions_logged"])
    c[1].metric("Results received", s["outcomes_in_window"])
    c[2].metric("Live accuracy", "—" if s["live_accuracy"] is None
                else f"{s['live_accuracy']*100:.1f}%")
    c[3].metric("Live log-loss", "—" if s["live_log_loss"] is None else f"{s['live_log_loss']:.3f}")
    st.subheader("Feature drift (PSI vs training data; above 0.2 = drift)")
    if s["psi"]:
        psi = pd.DataFrame({"feature": list(s["psi"]), "psi": list(s["psi"].values())})
        st.bar_chart(psi.set_index("feature"))
    else:
        st.caption(f"Needs more traffic ({s['drift_sample_size']} requests so far).")

st.subheader("Grafana dashboards")
dash = st.selectbox("Dashboard", ["acepredict-app", "acepredict-model", "acepredict-pipeline",
                                  "acepredict-infra"],
                    format_func={"acepredict-app": "Application", "acepredict-model":
                                 "Model quality & drift", "acepredict-pipeline": "Pipeline",
                                 "acepredict-infra": "Infrastructure"}.get)
components.iframe(f"{ui.GRAFANA_URL}/d/{dash}/?orgId=1&kiosk&refresh=5s", height=800,
                  scrolling=True)
ui.done("monitoring", t0)
