"""Run one Runpod Serverless job through the OAuth MCP runner; used by runpod_talking_head.

Usage: python _runpod_mcp_job.py <runpod_i2v.py> <request.json> <result.json>

request.json: {"endpoint_id": str, "input": dict, "timeout_s": int}
result.json:  the terminal get-job-status payload (output included). Nothing
base64 is printed, so job payloads never reach the terminal or logs.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


async def main(runner: Path, request_path: Path, result_path: Path) -> int:
    sys.path.insert(0, str(runner.parent))
    from runpod_i2v import Session, _call  # noqa: PLC0415

    req = json.loads(request_path.read_text())
    endpoint_id, timeout_s = req["endpoint_id"], int(req.get("timeout_s", 1800))
    async with Session() as session:
        sub = await _call(
            session,
            "run-endpoint",
            {"endpointId": endpoint_id, "input": req["input"], "policy": {"executionTimeout": timeout_s * 1000}},
        )
        job_id = sub.get("id") or sub.get("jobId")
        print(f"job {job_id}", file=sys.stderr, flush=True)
        deadline = time.time() + timeout_s
        status: dict = {"id": job_id, "status": "IN_QUEUE"}
        while time.time() < deadline:
            status = await _call(
                session, "get-job-status", {"endpointId": endpoint_id, "jobId": job_id, "wait": 45000}, timeout_s=90
            )
            print(f"{status.get('status')}", file=sys.stderr, flush=True)
            if status.get("status") in TERMINAL:
                break
        status.setdefault("id", job_id)
        result_path.write_text(json.dumps(status))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))))
