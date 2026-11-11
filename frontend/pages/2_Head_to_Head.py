"""Head-to-head record between two players."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import api_client as api
import ui

t0 = ui.page("Head to head", "⚔️", "h2h")
st.title("⚔️ Head to head")
tour = st.radio("Tour", ["atp", "wta"], horizontal=True, format_func=str.upper)
plist, err = ui.players(tour)
if err:
    st.error(err)
    st.stop()
by_id = {p["id"]: p for p in plist}
c1, c2 = st.columns(2)
p1 = c1.selectbox("Player 1", list(by_id), index=None, format_func=lambda i: ui.label(by_id[i]))
p2 = c2.selectbox("Player 2", [i for i in by_id if i != p1], index=None,
                  format_func=lambda i: ui.label(by_id[i]))
if p1 and p2:
    d, err = api.get("/h2h", p1=p1, p2=p2)
    if err:
        st.error(err)
        st.stop()
    st.markdown(f"<div class='ace-card'><div class='ace-big'>{d['p1_wins']} – {d['p2_wins']}"
                f"</div><div class='ace-muted'>{d['player1']['name']} vs "
                f"{d['player2']['name']}</div></div>", unsafe_allow_html=True)
    m = pd.DataFrame(d["meetings"])
    if m.empty:
        st.info("These players have never met in the data.")
    else:
        names = {d["player1"]["id"]: d["player1"]["name"], d["player2"]["id"]: d["player2"]["name"]}
        m["winner"] = m["winner_id"].map(names)
        st.dataframe(m[["date", "tournament", "surface", "round", "winner", "score"]],
                     hide_index=True, use_container_width=True)
ui.done("h2h", t0)
