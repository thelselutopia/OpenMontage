"""Contract coverage for the Runpod Serverless talking/singing head provider."""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import requests

from tools.avatar.runpod_talking_head import RunpodTalkingHead
from tools.base_tool import ToolStatus

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")

FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42fake-video"
_SUBPROCESS_RUN = subprocess.run


class _FakeResponse:
    def __init__(self, json_data):
        self._json = json_data

    def raise_for_status(self):
        return None

    def json(self):
        return self._json


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setenv("RUNPOD_TALKING_HEAD_ENDPOINT", "ep123")
    monkeypatch.setenv("RUNPOD_I2V_RUNNER", str(tmp_path / "missing_runner.py"))


@pytest.fixture()
def media(tmp_path):
    image = tmp_path / "face.png"
    audio = tmp_path / "line.wav"
    _SUBPROCESS_RUN(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=1376x768", "-frames:v", "1", str(image)],
        check=True,
    )
    _SUBPROCESS_RUN(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=220:d=3", str(audio)],
        check=True,
    )
    return image, audio


def test_discovered_under_avatar_capability():
    from tools.tool_registry import ToolRegistry

    registry = ToolRegistry()
    registry.discover()
    tool = registry.get("runpod_talking_head")
    assert tool is not None
    assert tool.capability == "avatar"
    assert tool.provider == "runpod"


def test_unavailable_without_endpoint_or_auth(monkeypatch):
    tool = RunpodTalkingHead()
    assert tool.get_status() == ToolStatus.UNAVAILABLE  # no API key, runner missing
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    assert tool.get_status() == ToolStatus.AVAILABLE
    monkeypatch.delenv("RUNPOD_TALKING_HEAD_ENDPOINT")
    assert tool.get_status() == ToolStatus.UNAVAILABLE


def test_size_derived_from_image_aspect(media):
    image, _ = media
    assert RunpodTalkingHead._derive_size(image, 832) == (832, 464)


def test_cost_scales_with_audio_duration(media):
    image, audio = media
    tool = RunpodTalkingHead()
    short = tool.estimate_cost({"image_path": str(image), "audio_path": str(audio), "audio_duration": 4})
    long = tool.estimate_cost({"image_path": str(image), "audio_path": str(audio), "audio_duration": 16})
    assert 0 < short < long


def test_padded_output_is_trimmed_to_audio(tmp_path):
    video = tmp_path / "padded.mp4"
    _SUBPROCESS_RUN(
        [
            "ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=320x176:r=25:d=3",
            "-f", "lavfi", "-i", "sine=f=220:d=3", "-shortest", "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
    )
    tool = RunpodTalkingHead()
    assert tool._trim_to_audio(video, 2.0) is True
    from tools.video._shared import probe_output

    assert probe_output(video)["duration_seconds"] == pytest.approx(2.0, abs=0.1)
    assert tool._trim_to_audio(video, 2.0) is False  # already within tolerance


def test_rest_transport_submits_and_decodes(monkeypatch, media, tmp_path):
    image, audio = media
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    calls = {}

    def fake_post(url, json, headers, timeout):
        calls["url"] = url
        calls["input"] = json["input"]
        return _FakeResponse({"id": "job-1"})

    def fake_get(url, headers, timeout):
        return _FakeResponse(
            {
                "id": "job-1",
                "status": "COMPLETED",
                "executionTime": 36000,
                "delayTime": 2000,
                "output": {"video": base64.b64encode(FAKE_MP4).decode()},
            }
        )

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)

    out = tmp_path / "out.mp4"
    result = RunpodTalkingHead().execute(
        {"image_path": str(image), "audio_path": str(audio), "prompt": "a woman singing", "output_path": str(out)}
    )

    assert result.success, result.error
    assert calls["url"].endswith("/ep123/run")
    assert calls["input"]["prompt"] == "a woman singing"
    assert (calls["input"]["width"], calls["input"]["height"]) == (832, 464)
    assert "max_frame" not in calls["input"]
    sent = base64.b64decode(calls["input"]["image_base64"])
    assert sent[:2] == b"\xff\xd8"  # re-encoded as JPEG at the output size, not the source PNG
    assert len(sent) < image.stat().st_size or len(sent) < 200_000
    assert out.read_bytes() == FAKE_MP4
    assert result.data["job_id"] == "job-1"
    assert result.data["transport"] == "rest"
    assert result.cost_usd == pytest.approx(36 / 3600 * 1.79, abs=1e-4)


def test_failed_job_reports_worker_error(monkeypatch, media, tmp_path):
    image, audio = media
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse({"id": "job-2"}))
    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **k: _FakeResponse({"id": "job-2", "status": "FAILED", "error": "no kernel image"}),
    )

    result = RunpodTalkingHead().execute(
        {"image_path": str(image), "audio_path": str(audio), "output_path": str(tmp_path / "x.mp4")}
    )

    assert not result.success
    assert "no kernel image" in result.error
    assert result.data["status"] == "FAILED"


def test_mcp_transport_uses_bridge(monkeypatch, media, tmp_path):
    image, audio = media
    runner = tmp_path / "runpod_i2v.py"
    runner.write_text("# stub")
    monkeypatch.setenv("RUNPOD_I2V_RUNNER", str(runner))
    seen = {}

    def fake_run(cmd, **kwargs):
        if cmd[0] != "uv":
            return _SUBPROCESS_RUN(cmd, **kwargs)
        request, result = Path(cmd[-2]), Path(cmd[-1])
        seen["request"] = json.loads(request.read_text())
        seen["runner"] = cmd[-3]
        result.write_text(
            json.dumps(
                {
                    "id": "job-3",
                    "status": "COMPLETED",
                    "executionTime": 1000,
                    "output": {"video": base64.b64encode(FAKE_MP4).decode()},
                }
            )
        )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    real_which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else real_which(name))
    monkeypatch.setattr(subprocess, "run", fake_run)

    out = tmp_path / "out.mp4"
    result = RunpodTalkingHead().execute(
        {"image_path": str(image), "audio_path": str(audio), "max_frame": 201, "output_path": str(out)}
    )

    assert result.success, result.error
    assert result.data["transport"] == "mcp_oauth"
    assert seen["runner"] == str(runner)
    assert seen["request"]["endpoint_id"] == "ep123"
    assert seen["request"]["input"]["max_frame"] == 201
    assert out.read_bytes() == FAKE_MP4
