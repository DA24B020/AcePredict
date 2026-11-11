"""Frontend Prometheus metrics on a side port (FRONTEND_METRICS_PORT, default 8502).

Imported modules are cached by Python, so these objects and the HTTP server are created
once per Streamlit process even though page scripts rerun on every interaction.
"""

from __future__ import annotations

import os

from prometheus_client import Counter, Histogram, start_http_server

PAGE_VIEWS = Counter("acepredict_frontend_page_views_total", "Page views", ["page"])
PREDICT_CLICKS = Counter("acepredict_frontend_predict_clicks_total", "Predict button clicks")
BACKEND_ERRORS = Counter("acepredict_frontend_backend_errors_total",
                         "Errors shown to users because a backend call failed")
RENDER_SECONDS = Histogram("acepredict_frontend_render_seconds", "Page render time", ["page"])

_started = False


def ensure_server() -> None:
    global _started
    if _started:
        return
    try:
        start_http_server(int(os.environ.get("FRONTEND_METRICS_PORT", "8502")))
    except OSError:
        pass  # already running in this process
    _started = True
