"""Predict page (home): pick two players and a surface, get win probabilities."""

from __future__ import annotations

import streamlit as st

import api_client as api
import ui
from metrics import PREDICT_CLICKS

t0 = ui.page("Predict a match", "🎾", "predict")
st.title("🎾 Who wins? Predict a tennis match")
st.write("Choose the tour, two players and the court surface. AcePredict estimates each "
         "player's chance of winning from their Elo ratings, form and head-to-head record.")

tour = st.radio("Tour", ["atp", "wta"], horizontal=True,
                format_func=lambda t: "ATP (men)" if t == "atp" else "WTA (women)")
plist, err = ui.players(tour)
if err:
    st.error(err)
    if st.button("Retry"):
        ui.players.clear()
        st.rerun()
    st.stop()
if not plist:
    st.info("No players available yet. Run the pipeline from the Pipeline console page.")
    st.stop()

by_id = {p["id"]: p for p in plist}
ids = list(by_id)
c1, c2 = st.columns(2)
with c1:
    p1 = st.selectbox("Player 1", ids, index=None, placeholder="Type a name…",
                      format_func=lambda i: ui.label(by_id[i]))
with c2:
    p2 = st.selectbox("Player 2", [i for i in ids if i != p1], index=None,
                      placeholder="Type a name…", format_func=lambda i: ui.label(by_id[i]))

c3, c4, c5, c6 = st.columns(4)
surface = c3.selectbox("Surface", ["Hard", "Clay", "Grass"])
level = c4.selectbox("Event", ["tour", "masters", "grand_slam"],
                     format_func={"tour": "Tour event", "masters": "Masters / 1000",
                                  "grand_slam": "Grand Slam"}.get)
default_bo = 1 if (level == "grand_slam" and tour == "atp") else 0
best_of = c5.selectbox("Best of", [3, 5], index=default_bo)
rnd = c6.selectbox("Round", ["early", "QF", "SF", "F"],
                   format_func={"early": "Early round", "QF": "Quarter-final",
                                "SF": "Semi-final", "F": "Final"}.get)

ready = p1 is not None and p2 is not None and p1 != p2
if not ready:
    st.caption("Pick two different players to enable the prediction.")
if st.button("Predict", type="primary", disabled=not ready):
    PREDICT_CLICKS.inc()
    with st.spinner("Asking the model…"):
        res, err = api.post("/predict", {"tour": tour, "player1_id": p1, "player2_id": p2,
                                         "surface": surface, "best_of": best_of,
                                         "tourney_level": level, "round": rnd})
    if err:
        st.error(err)
    else:
        st.session_state["last_prediction"] = res

res = st.session_state.get("last_prediction")
if res and res["player1"]["id"] in (p1, p2):
    a, b = res["player1"], res["player2"]
    pa, pb = res["p1_win_prob"], res["p2_win_prob"]
    k1, k2 = st.columns(2)
    for col, p, prob in ((k1, a, pa), (k2, b, pb)):
        col.markdown(f"<div class='ace-card'><div class='ace-muted'>{p['name']}</div>"
                     f"<div class='ace-big'>{prob*100:.1f}%</div>"
                     "<div class='ace-muted'>chance to win on "
                     f"{res['surface'].lower()}</div></div>",
                     unsafe_allow_html=True)
    st.progress(pa, text=f"{a['name']} {pa*100:.0f}%  ·  {b['name']} {pb*100:.0f}%")
    st.caption(f"Elo ratings alone would say {res['elo_only_p1_prob']*100:.0f}% for {a['name']}. "
               f"Model version {res['model_version']}.")
    st.subheader("Why?")
    for f in res["top_factors"]:
        who = a["name"] if f["favours"] == "player1" else b["name"]
        st.write(f"• **{f['label']}** favours **{who}**")
    h = res["head_to_head"]
    st.write(f"Head-to-head: {a['name']} {h['p1_wins']} – {h['p2_wins']} {b['name']}")
    for w in res["warnings"]:
        st.warning(w)

ui.done("predict", t0)
