"""Write the backend's OpenAPI spec to docs/openapi.json (the LLD endpoint reference)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "backend")]

from app.main import create_app  # noqa: E402

if __name__ == "__main__":
    out = ROOT / "docs" / "openapi.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(create_app().openapi(), indent=2))
    print(f"wrote {out}")
