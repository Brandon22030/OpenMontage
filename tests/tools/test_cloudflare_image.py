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

from tools.base_tool import Determinism, ToolStatus  # noqa: E402
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
        assert "seed" not in result.data
        assert "seed_note" not in result.data
        assert result.seed is None
        assert result.data["free_tier_note"]

        args, kwargs = mock_post.call_args
        assert args[0] == (
            f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}"
            "/ai/run/@cf/black-forest-labs/flux-1-schnell"
        )
        assert kwargs["headers"]["Authorization"] == f"Bearer {TOKEN}"
        assert kwargs["json"] == {"prompt": "a cat", "steps": 4}

    def test_steps_forwarded_seed_never_sent(self, tool, mock_post, tmp_path):
        # flux-1-schnell rejects any field besides prompt/steps (HTTP 400).
        mock_post.return_value = _ok()
        result = tool.execute(
            {"prompt": "a cat", "steps": 6, "seed": 42, "output_path": str(tmp_path / "a.jpg")}
        )
        assert result.success
        assert mock_post.call_args.kwargs["json"] == {"prompt": "a cat", "steps": 6}
        assert result.seed is None
        assert "seed" not in result.data
        assert result.data["seed_note"] == module._SEED_IGNORED_NOTE

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


class TestEdgeCases:
    def test_metadata_and_schema(self, tool):
        assert tool.name == "cloudflare_image"
        assert tool.version == "0.1.0"
        assert tool.dependencies == ["env:CLOUDFLARE_ACCOUNT_ID", "env:CLOUDFLARE_API_TOKEN"]
        assert tool.agent_skills == ["flux-best-practices"]
        assert tool.input_schema["required"] == ["prompt"]
        props = tool.input_schema["properties"]
        for forbidden in ("width", "height", "negative_prompt"):
            assert forbidden not in props
        assert props["steps"]["minimum"] == 1 and props["steps"]["maximum"] == 8
        assert props["prompt"]["maxLength"] == 2048
        assert tool.retry_policy.max_retries == 2
        assert tool.idempotency_key_fields == ["prompt", "model", "steps"]
        assert "seed" not in props
        assert tool.supports["seed"] is False
        assert tool.determinism == Determinism.STOCHASTIC
        assert tool.supports["negative_prompt"] is False
        assert "10,000" in tool.install_instructions
        assert "dash.cloudflare.com" in tool.install_instructions

    def test_timeout(self, tool, mock_post, tmp_path):
        mock_post.return_value = _ok()
        tool.execute({"prompt": "x", "output_path": str(tmp_path / "a.jpg")})
        assert mock_post.call_args.kwargs["timeout"] == 120

    def test_default_output_path_and_nested_parent(self, tool, mock_post, tmp_path, monkeypatch):
        mock_post.return_value = _ok()
        out = tmp_path / "a" / "b" / "c.jpeg"
        result = tool.execute({"prompt": "x", "output_path": str(out)})
        assert result.success and out.exists()
        assert result.data["format"] == "jpeg"
        monkeypatch.chdir(tmp_path)
        result = tool.execute({"prompt": "x"})
        assert result.success and (tmp_path / "cloudflare_image.jpg").exists()

    def test_uppercase_jpg_extension_raw_bytes(self, tool, mock_post, tmp_path):
        raw = _jpeg_bytes()
        mock_post.return_value = _ok(raw)
        out = tmp_path / "A.JPG"
        assert tool.execute({"prompt": "x", "output_path": str(out)}).success
        assert out.read_bytes() == raw

    def test_webp_conversion(self, tool, mock_post, tmp_path):
        from PIL import Image

        mock_post.return_value = _ok()
        out = tmp_path / "a.webp"
        result = tool.execute({"prompt": "x", "output_path": str(out)})
        assert result.success and result.data["format"] == "webp"
        with Image.open(out) as img:
            assert img.format == "WEBP"

    def test_unsupported_extension_fails_before_post(self, tool, mock_post, tmp_path):
        mock_post.return_value = _ok()
        result = tool.execute({"prompt": "x", "output_path": str(tmp_path / "a.xyz")})
        assert not result.success and result.cost_usd == 0.0
        assert "Unsupported output extension" in result.error
        mock_post.assert_not_called()

    @pytest.mark.parametrize("prompt", [None, "", "   ", 123, ["a"]])
    def test_invalid_prompt_fails_without_network(self, tool, mock_post, prompt):
        result = tool.execute({"prompt": prompt})
        assert not result.success and result.error == "prompt is required"
        mock_post.assert_not_called()

    def test_missing_prompt_key(self, tool, mock_post):
        result = tool.execute({})
        assert not result.success and result.error == "prompt is required"
        mock_post.assert_not_called()

    def test_prompt_length_limit(self, tool, mock_post, tmp_path):
        mock_post.return_value = _ok()
        out = tmp_path / "a.jpg"
        assert tool.execute({"prompt": "x" * 2048, "output_path": str(out)}).success
        mock_post.reset_mock()
        result = tool.execute({"prompt": "x" * 2049, "output_path": str(out)})
        assert not result.success and "2048" in result.error
        mock_post.assert_not_called()

    def test_cf_error_message_token_masked(self, tool, mock_post, tmp_path):
        mock_post.return_value = _FakeResponse(
            {"success": False, "errors": [{"message": f"echo {TOKEN}"}]}, status_code=400
        )
        result = tool.execute({"prompt": "x", "output_path": str(tmp_path / "a.jpg")})
        assert not result.success and TOKEN not in result.error and "***" in result.error

    def test_account_id_url_encoded(self, tool, mock_post, tmp_path, monkeypatch):
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a/b?c")
        mock_post.return_value = _ok()
        assert tool.execute({"prompt": "x", "output_path": str(tmp_path / "a.jpg")}).success
        assert "/accounts/a%2Fb%3Fc/ai/run/" in mock_post.call_args.args[0]

    def test_jpg_extension_with_png_bytes_converted(self, tool, mock_post, tmp_path):
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (1, 2, 3)).save(buf, format="PNG")
        mock_post.return_value = _ok(buf.getvalue())
        out = tmp_path / "a.jpg"
        result = tool.execute({"prompt": "x", "output_path": str(out)})
        assert result.success and result.data["format"] == "jpeg"
        assert out.read_bytes().startswith(b"\xff\xd8\xff")

    def test_non_image_bytes_with_png_fails(self, tool, mock_post, tmp_path):
        mock_post.return_value = _ok(b"not an image")
        result = tool.execute({"prompt": "x", "output_path": str(tmp_path / "a.png")})
        assert not result.success and result.cost_usd == 0.0
        assert TOKEN not in (result.error or "")

    @pytest.mark.parametrize("steps", [0, -1, 9, True, "4", 4.5])
    def test_invalid_steps_no_network(self, tool, mock_post, steps):
        result = tool.execute({"prompt": "x", "steps": steps})
        assert not result.success and result.cost_usd == 0.0
        mock_post.assert_not_called()

    @pytest.mark.parametrize("steps", [1, 8])
    def test_steps_boundaries(self, tool, mock_post, tmp_path, steps):
        mock_post.return_value = _ok()
        out = tmp_path / "a.jpg"
        assert tool.execute({"prompt": "x", "steps": steps, "output_path": str(out)}).success
        assert mock_post.call_args.kwargs["json"]["steps"] == steps

    def test_seed_zero_not_forwarded(self, tool, mock_post, tmp_path):
        mock_post.return_value = _ok()
        result = tool.execute({"prompt": "x", "seed": 0, "output_path": str(tmp_path / "a.jpg")})
        assert result.success
        assert "seed" not in mock_post.call_args.kwargs["json"]
        assert result.data["seed_note"] == module._SEED_IGNORED_NOTE

    @pytest.mark.parametrize("body", [None, [], "str", {"success": True}, {"success": True, "result": None},
                                      {"success": True, "result": {"image": ""}}])
    def test_malformed_bodies(self, tool, mock_post, tmp_path, body):
        mock_post.return_value = _FakeResponse(body)
        result = tool.execute({"prompt": "x", "output_path": str(tmp_path / "a.jpg")})
        assert not result.success and result.cost_usd == 0.0

    @pytest.mark.parametrize("status", [429, 401, 503])
    def test_429_and_http_errors_do_not_leak_token(self, tool, mock_post, tmp_path, status):
        mock_post.return_value = _FakeResponse(
            {"errors": [{"message": "bad"}]}, status_code=status
        )
        result = tool.execute({"prompt": "x", "output_path": str(tmp_path / "a.jpg")})
        assert not result.success and result.cost_usd == 0.0
        assert TOKEN not in (result.error or "")

    def test_empty_env_values_treated_missing(self, monkeypatch):
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "")
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "")
        result = CloudflareImage().execute({"prompt": "x"})
        assert not result.success and "CLOUDFLARE_API_TOKEN" in result.error
