"""FLUX.1 schnell image generation via Cloudflare Workers AI.

Free tier friendly: Cloudflare grants 10,000 neurons/day at no cost.
"""

from __future__ import annotations

import base64
import binascii
import os
import time
import urllib.parse
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolTier,
)

_DEFAULT_MODEL = "@cf/black-forest-labs/flux-1-schnell"
_API_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/" + _DEFAULT_MODEL
_RATE_LIMIT_MESSAGE = (
    "Cloudflare Workers AI quota or rate limit exceeded "
    "(free tier: 10,000 neurons/day, resets daily at 00:00 UTC)"
)
_FREE_TIER_NOTE = "Cloudflare Workers AI free tier: 10,000 neurons/day, resets daily at 00:00 UTC"
_JPEG_EXTENSIONS = {".jpg", ".jpeg"}
_JPEG_MAGIC = b"\xff\xd8\xff"
_MAX_PROMPT_LENGTH = 2048


class CloudflareImage(BaseTool):
    name = "cloudflare_image"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"
    provider = "cloudflare"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.SEEDED
    runtime = ToolRuntime.API

    dependencies = ["env:CLOUDFLARE_ACCOUNT_ID", "env:CLOUDFLARE_API_TOKEN"]
    install_instructions = (
        "Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN.\n"
        "  Find your Account ID and create an API token with the 'Workers AI' "
        "permission at https://dash.cloudflare.com\n"
        "  Free tier: 10,000 neurons/day."
    )
    agent_skills = ["flux-best-practices"]

    capabilities = ["generate_image", "text_to_image"]
    supports = {
        "seed": True,
        "negative_prompt": False,
        "custom_size": False,
        "image_edit": False,
    }
    best_for = [
        "free tier image generation (10,000 neurons/day)",
        "rapid prototyping and drafts",
        "FLUX schnell fast 1-8 step generation",
    ]
    not_good_for = [
        "custom image sizes",
        "precise text rendering in images",
        "offline generation",
    ]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string", "minLength": 1, "maxLength": 2048},
            "model": {
                "type": "string",
                "enum": [_DEFAULT_MODEL],
                "default": _DEFAULT_MODEL,
            },
            "steps": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4},
            "seed": {"type": "integer"},
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=100, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["prompt", "model", "steps", "seed"]
    side_effects = [
        "writes image file to output_path",
        "calls Cloudflare Workers AI API",
    ]
    user_visible_verification = ["Inspect generated image for relevance and quality"]

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        # Free tier: 10,000 neurons/day at no cost.
        return 0.0

    @staticmethod
    def _cf_errors(body: Any) -> str:
        """Extract Cloudflare error messages from a response body, if any."""
        if not isinstance(body, dict):
            return ""
        errors = body.get("errors")
        if not isinstance(errors, list):
            return ""
        messages = [
            str(e.get("message", e)) if isinstance(e, dict) else str(e) for e in errors
        ]
        return "; ".join(m for m in messages if m)

    @staticmethod
    def _fail(error: str) -> ToolResult:
        """Build a failed result, scrubbing the API token from any message."""
        token = os.environ.get("CLOUDFLARE_API_TOKEN")
        if token:
            error = error.replace(token, "***")
        return ToolResult(success=False, error=error, cost_usd=0.0)

    @staticmethod
    def _pillow_format(ext: str) -> str | None:
        """Return the Pillow format name for an extension, or None if unsupported."""
        from PIL import Image

        return Image.registered_extensions().get(ext)

    @staticmethod
    def _write_image(image_bytes: bytes, output_path: Path) -> str:
        """Write image bytes; keep raw JPEG bytes, otherwise convert with Pillow."""
        ext = output_path.suffix.lower()
        if ext in _JPEG_EXTENSIONS and image_bytes.startswith(_JPEG_MAGIC):
            output_path.write_bytes(image_bytes)
            return "jpeg"

        import io

        from PIL import Image

        fmt = Image.registered_extensions().get(ext)
        if fmt is None:
            raise ValueError(f"Unsupported output extension: {ext or '(none)'}")
        with Image.open(io.BytesIO(image_bytes)) as img:
            if fmt == "JPEG":
                img = img.convert("RGB")
            img.save(output_path, format=fmt)
        return fmt.lower()

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
        if not account_id or not api_token:
            return self._fail(
                "Cloudflare credentials not found. " + self.install_instructions
            )

        prompt = inputs.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return self._fail("prompt is required")
        if len(prompt) > _MAX_PROMPT_LENGTH:
            return self._fail(f"prompt must be at most {_MAX_PROMPT_LENGTH} characters")

        steps = inputs.get("steps", 4)
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 8:
            return self._fail(f"steps must be an integer between 1 and 8, got {steps!r}")

        output_path = Path(inputs.get("output_path", "cloudflare_image.jpg"))
        ext = output_path.suffix.lower()
        if ext not in _JPEG_EXTENSIONS and self._pillow_format(ext) is None:
            return self._fail(f"Unsupported output extension: {ext or '(none)'}")

        import requests

        start = time.time()
        seed = inputs.get("seed")

        payload: dict[str, Any] = {"prompt": prompt, "steps": steps}
        if seed is not None:
            payload["seed"] = seed

        try:
            response = requests.post(
                _API_URL.format(
                    account_id=urllib.parse.quote(account_id, safe="")
                ),
                headers={
                    "Authorization": f"Bearer {api_token}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )

            try:
                body = response.json()
            except Exception:
                body = None

            if response.status_code == 429:
                return self._fail(_RATE_LIMIT_MESSAGE)
            if response.status_code >= 400:
                detail = self._cf_errors(body)
                msg = f"Cloudflare Workers AI request failed (HTTP {response.status_code})"
                return self._fail(f"{msg}: {detail}" if detail else msg)

            if not isinstance(body, dict) or not body.get("success"):
                detail = self._cf_errors(body)
                return self._fail(
                    "Cloudflare Workers AI returned an unsuccessful response"
                    + (f": {detail}" if detail else "")
                )

            result = body.get("result")
            image_b64 = result.get("image") if isinstance(result, dict) else None
            if not image_b64:
                return self._fail("Cloudflare Workers AI response contained no image")

            try:
                image_bytes = base64.b64decode(image_b64, validate=True)
            except (binascii.Error, ValueError):
                return self._fail("Cloudflare Workers AI returned invalid base64 image data")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            fmt = self._write_image(image_bytes, output_path)
        except Exception as e:
            return self._fail(f"Cloudflare image generation failed: {e}")

        return ToolResult(
            success=True,
            data={
                "provider": "cloudflare",
                "model": _DEFAULT_MODEL,
                "prompt": prompt,
                "output": str(output_path),
                "format": fmt,
                "steps": steps,
                "seed": seed,
                "free_tier_note": _FREE_TIER_NOTE,
            },
            artifacts=[str(output_path)],
            cost_usd=0.0,
            duration_seconds=round(time.time() - start, 2),
            seed=seed,
            model=_DEFAULT_MODEL,
        )
