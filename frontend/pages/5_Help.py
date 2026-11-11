"""Short user guide and glossary."""

from __future__ import annotations

import streamlit as st

import ui

t0 = ui.page("Help", "❓", "help")
st.title("❓ How to use AcePredict")
st.markdown("""
**Get a prediction in 4 steps**
1. Open **Predict a match** (first page).
2. Choose **ATP** (men) or **WTA** (women).
3. Start typing each player's name and pick them from the list.
4. Choose the surface (and event / round if you know them) and press **Predict**.

**Reading the result**
- The two big numbers are each player's chance of winning. They always add up to 100%.
- *Elo ratings alone would say…* shows the simpler rating-only estimate for comparison.
- **Why?** lists the three factors that moved the prediction most.
- Yellow notes warn you when data is thin (few matches, long break, rare surface).

**Glossary**
- **Elo rating**: a skill score that goes up after wins and down after losses; beating a
  strong player gains more. 1500 is a new player; the best are above 2000.
- **Surface Elo**: the same score counted only on hard, clay or grass courts.
- **Form**: share of recent matches won.
- **Head-to-head**: how often these two players have beaten each other.

**Something went wrong?**
- *Service unreachable*: the backend is down. Ask the admin to run `./scripts/start.sh`.
- *No data yet / model not ready*: the pipeline has not run.
  Use **Pipeline console → Run pipeline**.

Full manual: `docs/user_manual.md`.
""")
ui.done("help", t0)
