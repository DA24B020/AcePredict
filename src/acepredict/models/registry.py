"""Model registry operations: promotion gate, promote, rollback, current champion.

Aliases (MLflow's replacement for stages):
  champion    the version the model server serves
  challenger  the version produced by the latest training run
  previous    the champion before the last promotion (rollback target)
"""

from __future__ import annotations

import logging
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

log = logging.getLogger(__name__)

CHAMPION, CHALLENGER, PREVIOUS = "champion", "challenger", "previous"


def get_alias_version(client: MlflowClient, name: str, alias: str):
    """Model version behind an alias, or None if the alias is not set."""
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def version_test_log_loss(client: MlflowClient, mv) -> float | None:
    """Test log-loss recorded on the training run of a model version."""
    if mv is None:
        return None
    tag = (mv.tags or {}).get("test_log_loss")
    if tag:
        return float(tag)
    metrics = client.get_run(mv.run_id).data.metrics
    return metrics.get("test_log_loss")


def promote(client: MlflowClient, name: str, version: str, reason: str = "manual") -> dict:
    """Make ``version`` the champion and remember the old champion as `previous`."""
    current = get_alias_version(client, name, CHAMPION)
    client.get_model_version(name, version)  # raises if the version does not exist
    if current is not None and str(current.version) != str(version):
        client.set_registered_model_alias(name, PREVIOUS, current.version)
    client.set_registered_model_alias(name, CHAMPION, version)
    client.set_model_version_tag(name, version, "promoted_reason", reason)
    log.info("promoted %s v%s to champion (%s)", name, version, reason)
    return {"champion_version": str(version),
            "previous_version": str(current.version) if current else None}


def rollback(client: MlflowClient, name: str) -> dict:
    """Swap champion and previous."""
    prev = get_alias_version(client, name, PREVIOUS)
    cur = get_alias_version(client, name, CHAMPION)
    if prev is None:
        raise MlflowException("no previous champion to roll back to")
    client.set_registered_model_alias(name, CHAMPION, prev.version)
    if cur is not None:
        client.set_registered_model_alias(name, PREVIOUS, cur.version)
    client.set_model_version_tag(name, prev.version, "promoted_reason", "rollback")
    log.info("rolled back %s to v%s", name, prev.version)
    return {"champion_version": str(prev.version),
            "previous_version": str(cur.version) if cur else None}


def promotion_gate(client: MlflowClient, name: str, challenger_version: str,
                   challenger_ll: float, tolerance: float) -> dict[str, Any]:
    """Promote the challenger only if its test log-loss is no worse than the champion's."""
    champion = get_alias_version(client, name, CHAMPION)
    champ_ll = version_test_log_loss(client, champion)
    decision: dict[str, Any] = {
        "challenger_version": str(challenger_version),
        "challenger_test_log_loss": challenger_ll,
        "champion_version_before": str(champion.version) if champion else None,
        "champion_test_log_loss": champ_ll,
    }
    if champion is None or champ_ll is None or challenger_ll <= champ_ll + tolerance:
        res = promote(client, name, challenger_version, reason="gate_passed")
        decision.update({"promoted": True, "champion_version_after": res["champion_version"],
                         "reason": "no champion yet" if champion is None else
                         "challenger log-loss <= champion log-loss"})
    else:
        decision.update({"promoted": False,
                         "champion_version_after": str(champion.version),
                         "reason": "challenger worse than champion; champion kept"})
    return decision
