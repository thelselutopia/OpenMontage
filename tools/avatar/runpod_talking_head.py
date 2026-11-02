"""Audio-driven talking/singing video on a Runpod Serverless worker.

Targets the Hub worker ``wlsdml1114/Infinitetalk_Runpod_hub`` (listed release
``skyreelsV3_v0.7_blackwell``), which runs SkyReels-V3 A2V in ComfyUI with
MelBandRoFormer vocal separation, so song stems with backing music work.

Worker gotchas (verified 2026-11-02):
- The release only runs on Blackwell GPUs (pool ``ADA_32_PRO`` / RTX 5090);
  on a 4090 SageAttention fails with "no kernel image is available".
- The stock handler returns the first video it finds, which is the 81-frame
  silent stage-1 reference. The endpoint must override its start command so
  the final node (131) wins:
    bash -c "sed -i 's/for node_id in videos:/for node_id in sorted(videos, key=lambda n: int(n) != 131):/' /handler.py && exec /entrypoint.sh"

Transports:
- ``RUNPOD_API_KEY`` set -> plain Runpod REST (/run + /status).
- otherwise -> the OAuth MCP runner at ``RUNPOD_I2V_RUNNER`` (default
  ~/Work/tools/runpod-i2v/runpod_i2v.py) via ``uv``, using ``_runpod_mcp_job.py``.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)
from tools.video._shared import probe_output

REST_BASE = "https://api.runpod.ai/v2"
DEFAULT_RUNNER = Path.home() / "Work" / "tools" / "runpod-i2v" / "runpod_i2v.py"
BRIDGE = Path(__file__).with_name("_runpod_mcp_job.py")

# RTX 5090 serverless rate from the Runpod catalog read on 2026-11-02.
PRICE_PER_HOUR = 1.79
# Measured: 8.36 s clip at 832x464 took 172.7 s of execution.
GPU_SECONDS_PER_VIDEO_SECOND = 21.0
COLD_START_SECONDS = 60.0
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


class RunpodTalkingHead(BaseTool):
    name = "runpod_talking_head"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "avatar"
    provider = "runpod"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = ["env:RUNPOD_TALKING_HEAD_ENDPOINT", "cmd:ffmpeg"]
    install_instructions = (
        "Deploy Hub repo wlsdml1114/Infinitetalk_Runpod_hub on gpuIds ADA_32_PRO, override its "
        "start command (see this module's docstring), and set RUNPOD_TALKING_HEAD_ENDPOINT in .env. "
        "Auth: RUNPOD_API_KEY, or the OAuth runner at RUNPOD_I2V_RUNNER with uv on PATH."
    )
    agent_skills = ["ffmpeg"]

    capabilities = ["photo_to_video", "face_animation", "audio_driven_animation", "singing_animation"]
    supports = {
        "photo_to_video": True,
        "audio_driven_animation": True,
        "music_with_vocals": True,
        "offline": False,
        "cloud_render": True,
    }
    best_for = [
        "close-up and medium shots of a character speaking or singing",
        "song vocals with backing music (vocals are separated on the worker)",
        "natural head, eye and expression motion beyond mouth-only lip sync",
    ]
    not_good_for = [
        "wide shots where the face is small",
        "single takes much longer than ~20 s (split at breaths instead)",
        "fully offline generation",
    ]
    fallback_tools = ["talking_head", "lip_sync"]

    input_schema = {
        "type": "object",
        "required": ["image_path", "audio_path"],
        "properties": {
            "image_path": {"type": "string", "description": "Character still"},
            "audio_path": {"type": "string", "description": "Driving speech or song audio"},
            "audio_start": {"type": "number", "description": "Seconds into audio_path to start"},
            "audio_duration": {"type": "number", "description": "Seconds of audio to use"},
            "prompt": {
                "type": "string",
                "default": "A person talking naturally",
                "description": "Scene/performance description, e.g. 'a woman singing softly into a vintage microphone'",
            },
            "width": {"type": "integer", "description": "Output width (multiple of 16); default from image aspect"},
            "height": {"type": "integer", "description": "Output height (multiple of 16); default from image aspect"},
            "long_side": {"type": "integer", "default": 832, "description": "Long side when width/height are derived"},
            "max_frame": {"type": "integer", "description": "Frame cap; default is derived from audio length"},
            "timeout_seconds": {"type": "integer", "default": 1800},
            "output_path": {"type": "string", "description": "Output video path (default: {image stem}_talking.mp4)"},
        },
    }

    resource_profile = ResourceProfile(cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=500, network_required=True)
    idempotency_key_fields = [
        "image_path",
        "audio_path",
        "audio_start",
        "audio_duration",
        "prompt",
        "width",
        "height",
        "max_frame",
    ]
    side_effects = ["paid Runpod Serverless GPU time", "writes video file to output_path"]
    user_visible_verification = [
        "Watch with sound for lip-sync drift, especially across pauses and breaths",
        "Check identity holds across the whole clip",
    ]
    latency_p50_seconds = 240.0

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> ToolStatus:
        if not os.environ.get("RUNPOD_TALKING_HEAD_ENDPOINT") or not shutil.which("ffmpeg"):
            return ToolStatus.UNAVAILABLE
        if os.environ.get("RUNPOD_API_KEY"):
            return ToolStatus.AVAILABLE
        if shutil.which("uv") and self._runner_path().is_file():
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    @staticmethod
    def _runner_path() -> Path:
        return Path(os.environ.get("RUNPOD_I2V_RUNNER") or DEFAULT_RUNNER).expanduser()

    # ------------------------------------------------------------------
    # Cost & runtime estimates
    # ------------------------------------------------------------------

    def _video_seconds(self, inputs: dict[str, Any]) -> float:
        if inputs.get("audio_duration"):
            return float(inputs["audio_duration"])
        audio = Path(inputs.get("audio_path", ""))
        if audio.is_file():
            probed = probe_output(audio)
            if probed.get("duration_seconds"):
                return float(probed["duration_seconds"]) - float(inputs.get("audio_start") or 0)
        return 10.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        return COLD_START_SECONDS + self._video_seconds(inputs) * GPU_SECONDS_PER_VIDEO_SECOND

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return round(self.estimate_runtime(inputs) / 3600 * PRICE_PER_HOUR, 4)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        image_path = Path(inputs["image_path"])
        audio_path = Path(inputs["audio_path"])
        if not image_path.is_file():
            return ToolResult(success=False, error=f"Image not found: {image_path}")
        if not audio_path.is_file():
            return ToolResult(success=False, error=f"Audio not found: {audio_path}")
        if self.get_status() != ToolStatus.AVAILABLE:
            return ToolResult(success=False, error="runpod_talking_head unavailable.\n" + self.install_instructions)

        endpoint_id = os.environ["RUNPOD_TALKING_HEAD_ENDPOINT"]
        output_path = Path(
            inputs.get("output_path") or image_path.with_name(f"{image_path.stem}_talking.mp4")
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()

        try:
            with tempfile.TemporaryDirectory(prefix="rp_talk_") as tmp:
                job_input = self._build_input(inputs, image_path, audio_path, Path(tmp))
                clip_seconds = probe_output(Path(tmp) / "audio.mp3").get("duration_seconds")
                timeout_s = int(inputs.get("timeout_seconds", 1800))
                if os.environ.get("RUNPOD_API_KEY"):
                    transport = "rest"
                    status = self._run_rest(endpoint_id, job_input, timeout_s)
                else:
                    transport = "mcp_oauth"
                    status = self._run_mcp(endpoint_id, job_input, timeout_s, Path(tmp))
        except Exception as exc:
            return ToolResult(success=False, data={"provider": self.provider}, error=f"Runpod talking head failed: {exc}")

        job_id = status.get("id")
        output = status.get("output") or {}
        if status.get("status") != "COMPLETED" or not isinstance(output, dict) or not output.get("video"):
            detail = status.get("error") or (output.get("error") if isinstance(output, dict) else None)
            return ToolResult(
                success=False,
                data={"provider": self.provider, "job_id": job_id, "status": status.get("status")},
                error=f"Runpod job {job_id} ended {status.get('status')}: {detail}",
            )

        output_path.write_bytes(base64.b64decode(output["video"]))
        trimmed = self._trim_to_audio(output_path, clip_seconds)
        probed = probe_output(output_path)
        execution_s = (status.get("executionTime") or 0) / 1000
        delay_s = (status.get("delayTime") or 0) / 1000

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": "skyreels-v3-a2v",
                "endpoint_id": endpoint_id,
                "job_id": job_id,
                "transport": transport,
                "image": str(image_path),
                "audio": str(audio_path),
                "prompt": job_input["prompt"],
                "width": job_input["width"],
                "height": job_input["height"],
                "output": str(output_path),
                "execution_seconds": execution_s,
                "queue_delay_seconds": delay_s,
                "trimmed_to_audio": trimmed,
                "format": "mp4",
                **probed,
            },
            artifacts=[str(output_path)],
            cost_usd=round(execution_s / 3600 * PRICE_PER_HOUR, 4) if execution_s else self.estimate_cost(inputs),
            duration_seconds=round(time.time() - start, 2),
            model="skyreels-v3-a2v",
        )

    def _build_input(self, inputs: dict[str, Any], image_path: Path, audio_path: Path, tmp: Path) -> dict[str, Any]:
        width, height = inputs.get("width"), inputs.get("height")
        if not (width and height):
            width, height = self._derive_size(image_path, int(inputs.get("long_side", 832)))
        if int(width) % 16 or int(height) % 16:
            raise ValueError("width and height must be multiples of 16")

        # Mono 44.1 kHz MP3 keeps the payload small without starving vocal separation.
        audio_clip = tmp / "audio.mp3"
        cmd = ["ffmpeg", "-loglevel", "error", "-y"]
        if inputs.get("audio_start"):
            cmd += ["-ss", str(inputs["audio_start"])]
        if inputs.get("audio_duration"):
            cmd += ["-t", str(inputs["audio_duration"])]
        cmd += ["-i", str(audio_path), "-ac", "1", "-ar", "44100", "-b:a", "160k", str(audio_clip)]
        self.run_command(cmd, timeout=120)

        # The worker crops/resizes to width x height anyway; sending a 2k PNG trips request-size
        # limits (the MCP route returns 413), so send a cover-cropped JPEG at the output size.
        image_clip = tmp / "image.jpg"
        w, h = int(width), int(height)
        self.run_command(
            [
                "ffmpeg", "-loglevel", "error", "-y", "-i", str(image_path), "-vf",
                f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h}",
                "-q:v", "2", str(image_clip),
            ],
            timeout=60,
        )

        job_input: dict[str, Any] = {
            "input_type": "image",
            "person_count": "single",
            "image_base64": base64.b64encode(image_clip.read_bytes()).decode(),
            "wav_base64": base64.b64encode(audio_clip.read_bytes()).decode(),
            "prompt": inputs.get("prompt") or "A person talking naturally",
            "width": int(width),
            "height": int(height),
        }
        if inputs.get("max_frame"):
            job_input["max_frame"] = int(inputs["max_frame"])
        return job_input

    def _trim_to_audio(self, video_path: Path, clip_seconds: float | None) -> bool:
        """The worker renders in ~81-frame windows and pads the last one; cut back to the audio."""
        duration = probe_output(video_path).get("duration_seconds")
        if not clip_seconds or not duration or duration <= clip_seconds + 0.1:
            return False
        trimmed = video_path.with_name(f"{video_path.stem}.trim{video_path.suffix}")
        self.run_command(
            [
                "ffmpeg", "-loglevel", "error", "-y", "-i", str(video_path), "-t", f"{clip_seconds:.3f}",
                "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", str(trimmed),
            ],
            timeout=300,
        )
        trimmed.replace(video_path)
        return True

    @staticmethod
    def _derive_size(image_path: Path, long_side: int) -> tuple[int, int]:
        probed = probe_output(image_path)
        w, h = probed.get("video_width"), probed.get("video_height")
        if not (w and h):
            return long_side, long_side
        scale = long_side / max(w, h)
        return max(16, round(w * scale / 16) * 16), max(16, round(h * scale / 16) * 16)

    def _run_rest(self, endpoint_id: str, job_input: dict[str, Any], timeout_s: int) -> dict[str, Any]:
        import requests

        headers = {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"}
        sub = requests.post(
            f"{REST_BASE}/{endpoint_id}/run",
            json={"input": job_input, "policy": {"executionTimeout": timeout_s * 1000}},
            headers=headers,
            timeout=120,
        )
        sub.raise_for_status()
        job_id = sub.json()["id"]
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            resp = requests.get(f"{REST_BASE}/{endpoint_id}/status/{job_id}", headers=headers, timeout=120)
            resp.raise_for_status()
            status = resp.json()
            if status.get("status") in TERMINAL:
                return status
            time.sleep(10)
        raise TimeoutError(f"job {job_id} not finished after {timeout_s}s")

    def _run_mcp(self, endpoint_id: str, job_input: dict[str, Any], timeout_s: int, tmp: Path) -> dict[str, Any]:
        request = tmp / "request.json"
        result = tmp / "result.json"
        request.write_text(json.dumps({"endpoint_id": endpoint_id, "input": job_input, "timeout_s": timeout_s}))
        cmd = [
            "uv", "run", "-q", "--with", "mcp>=1.10,<2", "--with", "pillow",
            "python", str(BRIDGE), str(self._runner_path()), str(request), str(result),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + 300, check=False)
        if proc.returncode != 0 or not result.is_file():
            raise RuntimeError(f"MCP bridge exited {proc.returncode}: {proc.stderr.strip()[-1500:]}")
        return json.loads(result.read_text())
