"""Tests for cloudflare_image (Cloudflare Workers AI FLUX.1 schnell). No real network."""

import base64
import io
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.base_tool import ToolStatus  # noqa: E402
from tools.graphics import cloudflare_image as module  # noqa: E402
from tools.graphics.cloudflare_image import CloudflareImage  # noqa: E402
from tools.tool_registry import ToolRegistry  # noqa: E402

ACCOUNT_ID = "acc123"
TOKEN = "secret-token-xyz"


def _jpeg_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 10, 10)).save(buf, format="JPEG")
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, json_data=None, status_code=200, raise_json=False):
        self._json_data = json_data
        self.status_code = status_code
        self._raise_json = raise_json

    def json(self):
        if self._raise_json:
            raise ValueError("not json")
        return self._json_data


def _ok(image_bytes: bytes | None = None) -> _FakeResponse:
    raw = image_bytes if image_bytes is not None else _jpeg_bytes()
    return _FakeResponse(
        {"success": True, "result": {"image": base64.b64encode(raw).decode()}}
    )


@pytest.fixture
def tool(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", ACCOUNT_ID)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", TOKEN)
    return CloudflareImage()


@pytest.fixture
def mock_post(monkeypatch):
    post = MagicMock()
    fake = types.ModuleType("requests")
    fake.post = post
    monkeypatch.setitem(sys.modules, "requests", fake)
    return post


class TestDiscovery:
    def test_discovered_in_registry(self):
        registry = ToolRegistry()
        registry.discover()
        found = registry.get("cloudflare_image")
        assert found is not None
        assert found.provider == "cloudflare"
        assert found.capability == "image_generation"

    def test_status_available_with_both_env(self, tool):
        assert tool.get_status() == ToolStatus.AVAILABLE

    @pytest.mark.parametrize("missing", ["CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"])
    def test_status_unavailable_when_env_missing(self, tool, monkeypatch, missing):
        monkeypatch.delenv(missing, raising=False)
        assert tool.get_status() == ToolStatus.UNAVAILABLE

    def test_provider_menu_placement(self, monkeypatch):
        registry = ToolRegistry()
        registry.discover()

        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", ACCOUNT_ID)
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", TOKEN)
        menu = registry.provider_menu()["image_generation"]
        assert "cloudflare_image" in [t["name"] for t in menu["available"]]

        monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
        menu = registry.provider_menu()["image_generation"]
        entry = next(t for t in menu["unavailable"] if t["name"] == "cloudflare_image")
        assert "CLOUDFLARE_API_TOKEN" in entry["install_instructions"]

    def test_estimate_cost_zero(self, tool):
        assert tool.estimate_cost({"prompt": "x"}) == 0.0
        assert tool.estimate_cost({"prompt": "x", "steps": 8}) == 0.0

    def test_schema_default_matches_constant(self):
        default = CloudflareImage.input_schema["properties"]["model"]["default"]
        assert default == module._DEFAULT_MODEL


class TestExecuteSuccess:
    def test_success_jpg(self, tool, mock_post, tmp_path):
        raw = _jpeg_bytes()
        mock_post.return_value = _ok(raw)
        out = tmp_path / "sub" / "img.jpg"

        result = tool.execute({"prompt": "a cat", "output_path": str(out)})

        assert result.success
        assert result.cost_usd == 0.0
        assert result.artifacts == [str(out)]
        assert out.read_bytes() == raw
        assert result.data["format"] == "jpeg"
        assert result.data["steps"] == 4
        assert result.data["seed"] is None
        assert result.data["free_tier_note"]

        args, kwargs = mock_post.call_args
        assert args[0] == (
            f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}"
            "/ai/run/@cf/black-forest-labs/flux-1-schnell"
        )
        assert kwargs["headers"]["Authorization"] == f"Bearer {TOKEN}"
        assert kwargs["json"] == {"prompt": "a cat", "steps": 4}

    def test_seed_and_steps_forwarded(self, tool, mock_post, tmp_path):
        mock_post.return_value = _ok()
        result = tool.execute(
            {"prompt": "a cat", "steps": 6, "seed": 42, "output_path": str(tmp_path / "a.jpg")}
        )
        assert result.success
        assert mock_post.call_args.kwargs["json"] == {"prompt": "a cat", "steps": 6, "seed": 42}
        assert result.data["seed"] == 42

    def test_png_conversion(self, tool, mock_post, tmp_path):
        from PIL import Image

        mock_post.return_value = _ok()
        out = tmp_path / "img.png"
        result = tool.execute({"prompt": "a cat", "output_path": str(out)})

        assert result.success
        assert result.data["format"] == "png"
        with Image.open(out) as img:
            assert img.format == "PNG"


class TestExecuteErrors:
    def _assert_failed(self, result):
        assert not result.success
        assert result.cost_usd == 0.0
        assert TOKEN not in (result.error or "")

    def test_missing_env(self, monkeypatch):
        monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
        monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
        result = CloudflareImage().execute({"prompt": "x"})
        self._assert_failed(result)
        assert "CLOUDFLARE_API_TOKEN" in result.error

    @pytest.mark.parametrize("steps", [0, 9])
    def test_steps_out_of_range(self, tool, mock_post, steps):
        result = tool.execute({"prompt": "x", "steps": steps})
        self._assert_failed(result)
        mock_post.assert_not_called()

    def test_rate_limit(self, tool, mock_post, tmp_path):
        mock_post.return_value = _FakeResponse({}, status_code=429)
        result = tool.execute({"prompt": "x", "output_path": str(tmp_path / "a.jpg")})
        self._assert_failed(result)
        assert "10,000 neurons/day" in result.error

    def test_http_error_with_cf_errors(self, tool, mock_post, tmp_path):
        mock_post.return_value = _FakeResponse(
            {"success": False, "errors": [{"code": 1, "message": "boom upstream"}]},
            status_code=500,
        )
        result = tool.execute({"prompt": "x", "output_path": str(tmp_path / "a.jpg")})
        self._assert_failed(result)
        assert "500" in result.error
        assert "boom upstream" in result.error

    def test_http_error_non_json(self, tool, mock_post, tmp_path):
        mock_post.return_value = _FakeResponse(status_code=502, raise_json=True)
        result = tool.execute({"prompt": "x", "output_path": str(tmp_path / "a.jpg")})
        self._assert_failed(result)
        assert "502" in result.error

    def test_success_false(self, tool, mock_post, tmp_path):
        mock_post.return_value = _FakeResponse(
            {"success": False, "errors": [{"message": "bad prompt"}]}
        )
        result = tool.execute({"prompt": "x", "output_path": str(tmp_path / "a.jpg")})
        self._assert_failed(result)
        assert "bad prompt" in result.error

    def test_missing_image(self, tool, mock_post, tmp_path):
        mock_post.return_value = _FakeResponse({"success": True, "result": {}})
        result = tool.execute({"prompt": "x", "output_path": str(tmp_path / "a.jpg")})
        self._assert_failed(result)
        assert "no image" in result.error

    def test_invalid_base64(self, tool, mock_post, tmp_path):
        mock_post.return_value = _FakeResponse(
            {"success": True, "result": {"image": "!!!not-base64!!!"}}
        )
        result = tool.execute({"prompt": "x", "output_path": str(tmp_path / "a.jpg")})
        self._assert_failed(result)
        assert "base64" in result.error

    def test_transport_exception_does_not_leak_token(self, tool, mock_post, tmp_path):
        mock_post.side_effect = RuntimeError(f"connection died using {TOKEN}")
        result = tool.execute({"prompt": "x", "output_path": str(tmp_path / "a.jpg")})
        self._assert_failed(result)


class TestNetworkGuard:
    def test_unmocked_call_fails_gracefully(self, tool, tmp_path):
        result = tool.execute({"prompt": "x", "output_path": str(tmp_path / "a.jpg")})
        assert not result.success
        assert result.cost_usd == 0.0
