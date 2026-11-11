"""Shared page setup: theme tweaks, sidebar status, cached player lists."""

from __future__ import annotations

import os
import time

import requests
import streamlit as st

import api_client as api
from metrics import PAGE_VIEWS, RENDER_SECONDS, ensure_server

GRAFANA_URL = os.environ.get("GRAFANA_PUBLIC_URL", "http://localhost:3000")
MLFLOW_URL = os.environ.get("MLFLOW_PUBLIC_URL", "http://localhost:5000")
PROMETHEUS_URL = os.environ.get("PROMETHEUS_PUBLIC_URL", "http://localhost:9090")
RAY_URL = os.environ.get("RAY_PUBLIC_URL", "http://localhost:8265")

CSS = """
<style>
.ace-card {background: #ffffff; border: 1px solid #dfe6da; border-radius: 12px;
           padding: 1rem 1.2rem; margin-bottom: .8rem;}
.ace-big {font-size: 2.4rem; font-weight: 700; color: #1f5f2c; line-height: 1.1;}
.ace-muted {color: #5d6b60; font-size: .9rem;}
</style>
"""


def page(title: str, icon: str, key: str) -> float:
    st.set_page_config(page_title=f"AcePredict · {title}", page_icon=icon, layout="wide")
    ensure_server()
    PAGE_VIEWS.labels(key).inc()
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### 🎾 AcePredict")
        try:
            r = requests.get(f"{api.BACKEND_URL}/ready", timeout=5).json()
        except (requests.RequestException, ValueError):
            st.error("Service unreachable")
        else:
            if r.get("ready"):
                st.success(f"Model v{r.get('model_version')} · data to {r.get('snapshot_date')}")
            else:
                st.warning(f"Not ready · data loaded: {r.get('snapshot_loaded')} · "
                           f"model server: {r.get('model_server')}")
        st.caption("Data: Jeff Sackmann's tennis_atp / tennis_wta (CC BY-NC-SA 4.0)")
    return time.perf_counter()


def done(key: str, t0: float) -> None:
    RENDER_SECONDS.labels(key).observe(time.perf_counter() - t0)


@st.cache_data(ttl=300, show_spinner=False)
def players(tour: str) -> tuple[list[dict], str | None]:
    data, err = api.get("/players", tour=tour, limit=1500, active_only=True)
    return (data or []), err


def label(p: dict) -> str:
    rank = f"#{p['rank']}" if p.get("rank") else "unranked"
    return f"{p['name']}  ({p.get('country') or '—'}, {rank}, Elo {p['elo']:.0f})"
