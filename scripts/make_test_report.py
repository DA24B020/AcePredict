"""Turn a pytest JUnit XML file into docs/test_report_unit.md (counts + per-test results).

    pytest --junitxml=reports/junit.xml ; python scripts/make_test_report.py reports/junit.xml
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(xml_path: str) -> int:
    root = ET.parse(xml_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    rows, totals = [], {"passed": 0, "failed": 0, "skipped": 0}
    for suite in suites:
        for case in suite.iter("testcase"):
            if case.find("failure") is not None or case.find("error") is not None:
                status = "failed"
            elif case.find("skipped") is not None:
                status = "skipped"
            else:
                status = "passed"
            totals[status] += 1
            rows.append((case.get("classname", ""), case.get("name", ""), status,
                         float(case.get("time", 0))))
    total = sum(totals.values())
    lines = [
        "# AcePredict — Test Report", "",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M UTC} from `{Path(xml_path).name}`.", "",
        f"**Total {total} · passed {totals['passed']} · failed {totals['failed']} · "
        f"skipped {totals['skipped']}**", "",
    ]
    load = ROOT / "reports" / "load_test.json"
    if load.exists():
        lt = json.loads(load.read_text())
        lines += ["## Load test", "",
                  f"{lt['requests']} requests in {lt['seconds']} s with {lt['concurrency']} "
                  f"clients: **{lt['throughput_rps']} req/s**, p50 {lt['p50_ms']} ms, "
                  f"p95 {lt['p95_ms']} ms (limit {lt['p95_limit_ms']} ms) → "
                  f"{'PASS' if lt['passed'] else 'FAIL'}", ""]
    lines += ["## Test cases", "", "| Module | Test | Result | Time (s) |", "|---|---|---|---|"]
    lines += [f"| {c} | {n} | {s} | {t:.2f} |" for c, n, s, t in rows]
    out = ROOT / "docs" / "test_report_unit.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}: {totals}")
    return 0 if totals["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "reports" / "junit.xml")))
