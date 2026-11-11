"""Player profile: Elo history by surface and recent results."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import api_client as api
import ui

t0 = ui.page("Player profile", "👤", "player")
st.title("👤 Player profile")
tour = st.radio("Tour", ["atp", "wta"], horizontal=True, format_func=str.upper)
plist, err = ui.players(tour)
if err:
    st.error(err)
    st.stop()
by_id = {p["id"]: p for p in plist}
pid = st.selectbox("Player", list(by_id), index=None, placeholder="Type a name…",
                   format_func=lambda i: ui.label(by_id[i]))
if pid:
    with st.spinner("Loading…"):
        d, err = api.get(f"/players/{pid}")
    if err:
        st.error(err)
        st.stop()
    c = st.columns(5)
    c[0].metric("Elo", f"{d['elo']:.0f}")
    c[1].metric("Hard / Clay / Grass", f"{d['elo_hard']:.0f} / {d['elo_clay']:.0f} / "
                                       f"{d['elo_grass']:.0f}")
    c[2].metric("Rank", d["rank"] or "—")
    c[3].metric("Record", f"{d['wins']}–{d['losses']}")
    c[4].metric("Form (last 10)", f"{(d['form_last10'] or 0)*100:.0f}%")
    hist = pd.DataFrame(d["elo_history"])
    if not hist.empty:
        st.subheader("Elo rating over time")
        st.line_chart(hist.set_index("month")[["elo", "elo_hard", "elo_clay", "elo_grass"]])
    st.subheader("Recent matches")
    rm = pd.DataFrame(d["recent_matches"])
    if not rm.empty:
        rm["result"] = rm["won"].map({True: "Won", False: "Lost"})
        st.dataframe(rm[["date", "tournament", "surface", "round", "opponent", "result", "score"]],
                     hide_index=True, use_container_width=True)
ui.done("player", t0)
