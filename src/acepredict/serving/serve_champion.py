"""Model-server supervisor: serve `models:/<name>@champion` with `mlflow models serve`.

It polls the registry; when the champion alias moves (promotion, rollback, retraining)
it restarts the MLflow scoring server on the new version. Until a champion exists it
waits, so the stack can start before the first pipeline run.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time

from mlflow.tracking import MlflowClient

from acepredict.logging_conf import setup_logging
from acepredict.models.registry import CHAMPION, get_alias_version

log = logging.getLogger("acepredict.serving")


def serve_command(name: str, host: str, port: int, workers: int) -> list[str]:
    return ["mlflow", "models", "serve", "-m", f"models:/{name}@{CHAMPION}", "-h", host,
            "-p", str(port), "--env-manager", "local", "--workers", str(workers)]


def main() -> None:
    setup_logging(os.environ.get("LOG_LEVEL", "INFO"))
    name = os.environ.get("MODEL_NAME", "acepredict")
    host = os.environ.get("MODEL_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("MODEL_SERVER_PORT", "5001"))
    workers = int(os.environ.get("MODEL_SERVER_WORKERS", "2"))
    poll = float(os.environ.get("CHAMPION_POLL_SECONDS", "30"))
    client = MlflowClient()
    proc: subprocess.Popen | None = None
    serving: str | None = None
    stopping = False

    def _stop(*_: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    while not stopping:
        try:
            mv = get_alias_version(client, name, CHAMPION)
        except Exception as exc:  # registry temporarily unreachable: keep serving as is
            log.warning("registry not reachable: %s", exc)
            time.sleep(poll)
            continue
        version = str(mv.version) if mv is not None else None
        if version is None and serving is None:
            log.info("no champion registered yet for '%s'; waiting (run the pipeline)", name)
        elif version != serving or (proc is not None and proc.poll() is not None):
            if proc is not None and proc.poll() is None:
                log.info("champion changed %s -> %s; restarting model server", serving, version)
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
            log.info("serving %s version %s on %s:%d", name, version, host, port)
            proc = subprocess.Popen(serve_command(name, host, port, workers),
                                    stdout=sys.stdout, stderr=sys.stderr)
            serving = version
        time.sleep(poll)
    if proc is not None and proc.poll() is None:
        proc.terminate()


if __name__ == "__main__":
    main()
