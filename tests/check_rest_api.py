#!/usr/bin/env python3
"""
Integration smoke test for the QQQ-hedge plain-HTTP REST API (async job pattern).

Exercises POST /api/hedge (submit) + GET /api/hedge/{job_id} (poll) plus error
paths. Hits a LIVE server over the network -- stdlib only, no deps.

Usage:
    python3 tests/check_rest_api.py [BASE_URL]
    QQQ_HEDGE_URL=http://localhost:8000 python3 tests/check_rest_api.py

Default BASE_URL: https://qqq-hedge-production.up.railway.app
Named so pytest does NOT auto-collect it (no test_ prefix / _test suffix).
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = (
    (sys.argv[1] if len(sys.argv) > 1 else None)
    or os.environ.get("QQQ_HEDGE_URL")
    or "https://qqq-hedge-production.up.railway.app"
).rstrip("/")

TIMEOUT = 90
POLL_MAX = 30      # poll attempts
POLL_EVERY = 2.0   # seconds


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def http(method, path, body=None):
    """Return (status_code, parsed_json_or_text)."""
    headers = {"Content-Type": "application/json"}
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            raw, code = r.read().decode(), r.status
    except urllib.error.HTTPError as e:
        raw, code = e.read().decode(), e.code
    try:
        return code, json.loads(raw)
    except Exception:
        return code, raw


def submit_and_wait(payload):
    """POST a job (assert 202), poll until terminal. Returns (submit_json, final_json)."""
    code, sub = http("POST", "/api/hedge", payload)
    assert code == 202, f"expected 202 on submit, got {code}: {sub}"
    assert "job_id" in sub and sub.get("status") == "pending", f"bad submit body: {sub}"
    assert sub.get("poll_url") == f"/api/hedge/{sub['job_id']}", f"bad poll_url: {sub}"

    jid = sub["job_id"]
    for _ in range(POLL_MAX):
        code, job = http("GET", f"/api/hedge/{jid}")
        status = job.get("status")
        if status in ("pending", "running"):
            assert code == 202, f"expected 202 while {status}, got {code}"
        elif status in ("done", "error"):
            assert code == 200, f"expected 200 when {status}, got {code}"
            return sub, job
        else:
            raise AssertionError(f"unexpected status {status!r}: {job}")
        time.sleep(POLL_EVERY)
    raise AssertionError(f"job {jid} did not finish within {POLL_MAX} polls")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_health():
    code, body = http("GET", "/health")
    assert code == 200, f"health {code}"
    assert isinstance(body, dict) and body.get("status") == "ok", body


def check_submit_default():
    """No args -> latest trading day, VT15."""
    _, job = submit_and_wait({})
    assert job["status"] == "done", job
    r = job["result"]
    assert r["vt"] == 15, r
    assert 0.0 <= r["exposure"] <= 1.5, r
    assert r["gate"] in (0.0, 0.5, 1.0), r
    # exposure reconciles with gate x w_vol
    assert abs(r["exposure"] - r["gate"] * r["w_vol"]) < 1e-6, r


def check_submit_date_and_vt():
    """date + vt in body."""
    _, job = submit_and_wait({"date": "2026-05-28", "vt": 23})
    assert job["status"] == "done", job
    r = job["result"]
    assert r["vt"] == 23 and abs(r["target_vol"] - 0.23) < 1e-9, r
    assert r["as_of_date"] <= "2026-05-28", r
    expected_w = min(0.23 / r["rv20"], 1.5)
    assert abs(r["w_vol"] - expected_w) < 1e-3, r


def check_submit_via_query_params():
    """Empty body, params in the query string."""
    code, sub = http("POST", "/api/hedge?vt=10", None)
    assert code == 202, f"{code}: {sub}"
    jid = sub["job_id"]
    job = None
    for _ in range(POLL_MAX):
        c, job = http("GET", f"/api/hedge/{jid}")
        if job.get("status") in ("done", "error"):
            break
        time.sleep(POLL_EVERY)
    assert job["status"] == "done", job
    assert job["result"]["vt"] == 10, job


def check_weekend_rolls_back():
    """A non-trading day rolls back to the prior trading day."""
    _, job = submit_and_wait({"date": "2026-05-24", "vt": 15})  # Sunday
    assert job["status"] == "done", job
    assert job["result"]["as_of_date"] < "2026-05-24", job["result"]


def check_error_invalid_vt():
    code, body = http("POST", "/api/hedge", {"vt": "not-a-number"})
    assert code == 400, f"expected 400, got {code}: {body}"


def check_error_nonpositive_vt():
    for v in (0, -5):
        code, body = http("POST", "/api/hedge", {"vt": v})
        assert code == 400, f"vt={v} expected 400, got {code}: {body}"


def check_error_unknown_job():
    code, body = http("GET", "/api/hedge/does-not-exist-12345")
    assert code == 404, f"expected 404, got {code}: {body}"
    assert isinstance(body, dict) and "error" in body, body


def check_error_date_before_data():
    """Job runs but the computation returns an error -> status 'error', HTTP 200."""
    _, job = submit_and_wait({"date": "2015-01-01", "vt": 15})
    assert job["status"] == "error", job
    assert "error" in job and job["error"], job


def check_book_vol_signal():
    """book_id -> portfolio vol source; gate must still match the QQQ signal."""
    code, sub = http("POST", "/api/hedge", {"book_id": 31, "vt": 15})
    assert code == 202, f"{code}: {sub}"
    jid = sub["job_id"]
    job = None
    for _ in range(75):          # book path does N+1 AV calls -> allow ~150s
        c, job = http("GET", f"/api/hedge/{jid}")
        if job.get("status") in ("done", "error"):
            break
        time.sleep(2)
    assert job["status"] == "done", job
    r = job["result"]
    assert r["vol_source"] == "portfolio", r
    assert r.get("portfolio_vol", 0) > 0, r
    assert r.get("book_name"), r
    assert r.get("n_constituents", 0) > 0, r
    assert abs(r["exposure"] - r["gate"] * r["w_vol"]) < 1e-6, r
    assert abs(r["w_vol"] - min(15 / 100 / r["rv20"], 1.5)) < 1e-3, r
    # Decoupling: the gate must equal the plain QQQ signal for the same date.
    _, qjob = submit_and_wait({"date": r["as_of_date"], "vt": 15})
    assert qjob["result"]["gate"] == r["gate"], "book gate must match QQQ gate"


def check_book_unknown():
    _, job = submit_and_wait({"book_id": 99999999, "vt": 15})
    assert job["status"] == "error", job


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

CHECKS = [
    check_health,
    check_submit_default,
    check_submit_date_and_vt,
    check_submit_via_query_params,
    check_weekend_rolls_back,
    check_error_invalid_vt,
    check_error_nonpositive_vt,
    check_error_unknown_job,
    check_error_date_before_data,
    check_book_vol_signal,
    check_book_unknown,
]


def main():
    print(f"Testing REST API at: {BASE}\n")
    passed = failed = 0
    for fn in CHECKS:
        name = fn.__name__
        t0 = time.time()
        try:
            fn()
            dt = time.time() - t0
            print(f"PASS  {name}  ({dt:.1f}s)")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
