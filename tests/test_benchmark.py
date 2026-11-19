import asyncio
import base64
import io
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from PIL import Image

from floorplanbench import client as client_module
from floorplanbench.client import RunConfig, digest, endpoint_url, prepare_image
from floorplanbench.render import plot_plan
from floorplanbench.schema import (
    DEFAULT_PROMPT,
    floorplan_json_schema,
    parse_plan,
    renderable,
    room_polygon,
)
from floorplanbench.service import run_model
from floorplanbench.storage import Store

from .samples import EXAMPLE, example_image


@pytest.fixture
def endpoint():
    state = {"requests": [], "status": 200, "text": json.dumps(EXAMPLE), "finish": "stop"}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(
                {"path": self.path, "payload": payload, "auth": self.headers.get("Authorization")}
            )
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("x-request-id", "req-local-test")
            self.end_headers()
            if state["status"] != 200:
                body = {"error": {"message": "secret-test-key DO NOT EXPOSE", "type": "api_error"}}
            elif self.path.endswith("/responses"):
                body = {
                    "id": "resp-test",
                    "created_at": 1,
                    "object": "response",
                    "model": "test-vlm-snapshot",
                    "status": state.get("response_status", "completed"),
                    "incomplete_details": state.get("incomplete_details"),
                    "error": state.get("response_error"),
                    "parallel_tool_calls": False,
                    "output": state.get(
                        "output",
                        [
                            {
                                "id": "msg-test",
                                "type": "message",
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": state["text"],
                                        "annotations": [],
                                    }
                                ],
                            }
                        ],
                    ),
                    "usage": {
                        "input_tokens": 30,
                        "output_tokens": 50,
                        "total_tokens": 80,
                        "input_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 5},
                        "output_tokens_details": {"reasoning_tokens": 7},
                    },
                }
            else:
                body = {
                    "id": "test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test-vlm",
                    "system_fingerprint": "fp-test",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": state["text"],
                                "refusal": state.get("refusal"),
                            },
                            "finish_reason": state["finish"],
                        }
                    ],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 50, "total_tokens": 80},
                }
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}/v1"
    yield state
    server.shutdown()
    server.server_close()
    thread.join()


def config(endpoint):
    return RunConfig(base_url=endpoint["url"], model="test-vlm", prompt=DEFAULT_PROMPT)


def run(store, cfg, session="session-one"):
    return asyncio.run(run_model(store, session, cfg, "secret-test-key", example_image()))


def test_real_openai_sdk_multimodal_roundtrip(endpoint, tmp_path):
    store = Store(tmp_path / "runs.sqlite3")
    cfg = replace(
        config(endpoint),
        token_parameter="max_completion_tokens",
        temperature=None,
        response_format="json_object",
        seed=42,
        image_detail="high",
    )
    result = run(store, cfg)
    request = endpoint["requests"][0]
    assert request["path"] == "/v1/chat/completions"
    assert request["auth"] == "Bearer secret-test-key"
    payload = request["payload"]
    assert payload["messages"][0] == {"role": "system", "content": DEFAULT_PROMPT}
    assert "temperature" not in payload and "max_tokens" not in payload
    assert payload["max_completion_tokens"] == 32768
    assert payload["seed"] == 42 and payload["response_format"] == {"type": "json_object"}
    image_block = payload["messages"][1]["content"][0]["image_url"]
    actual_image = base64.b64decode(image_block["url"].split(",", 1)[1])
    assert digest(actual_image) == result["image_sha256"]
    assert image_block["detail"] == "high"
    assert result["success"] and result["usage"]["total_tokens"] == 80
    assert result["response"]["request_id"] == "req-local-test"
    assert result["response"]["system_fingerprint"] == "fp-test"
    assert result["response"]["response_id"] == "test"
    loaded, image_bytes = store.get(result["id"], "session-one")
    assert loaded == result and image_bytes == actual_image
    assert "secret-test-key" not in json.dumps(loaded)
    assert plot_plan(parse_plan(result["raw"])[0]).layout.yaxis.scaleanchor == "x"


def test_official_responses_defaults_roundtrip(endpoint, tmp_path, monkeypatch):
    sdk = client_module.AsyncOpenAI
    clients = []

    def local_client(**kwargs):
        assert kwargs["base_url"] == "https://api.openai.com/v1"
        assert kwargs["max_retries"] == 0
        instance = sdk(**(kwargs | {"base_url": endpoint["url"]}))
        clients.append(instance)
        return instance

    monkeypatch.setattr(client_module, "AsyncOpenAI", local_client)
    cfg = RunConfig("https://api.openai.com/v1", "gpt-6-astra", DEFAULT_PROMPT)
    store = Store(tmp_path / "runs.sqlite3")
    result = run(store, cfg)
    assert result["success"] and result["api"] == "responses"
    request = endpoint["requests"][0]
    assert request["path"] == "/v1/responses"
    payload = request["payload"]
    assert payload["store"] is False and payload["max_output_tokens"] == 32768
    assert (
        not {"max_tokens", "max_completion_tokens", "temperature", "top_p", "seed"} & payload.keys()
    )
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True
    assert payload["input"][0] == {"role": "system", "content": DEFAULT_PROMPT}
    image = payload["input"][1]["content"][0]
    assert image["type"] == "input_image" and image["detail"] == "auto"
    sent = base64.b64decode(image["image_url"].split(",", 1)[1])
    assert digest(sent) == result["image_sha256"]
    assert result["request_parameters"]["text"] == payload["text"]
    assert result["response"]["model"] == "test-vlm-snapshot"
    assert result["response"]["response_id"] == "resp-test"
    assert result["response"]["request_id"] == "req-local-test"
    assert result["finish_reason"] == "completed"
    assert result["usage"]["input_tokens_details"] == {"cached_tokens": 10, "cache_write_tokens": 5}
    assert result["usage"]["output_tokens_details"]["reasoning_tokens"] == 7
    assert store.get(result["id"], "session-one")[0] == result
    assert "secret-test-key" not in json.dumps(result)
    assert clients[0].is_closed()


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
def test_explicit_third_party_schema_and_reasoning(endpoint, tmp_path, api):
    cfg = replace(
        config(endpoint), api_mode=api, response_format="json_schema", reasoning_effort="xhigh"
    )
    result = run(Store(tmp_path / "runs.sqlite3"), cfg)
    assert result["success"]
    payload = endpoint["requests"][0]["payload"]
    if api == "responses":
        assert payload["reasoning"] == {"effort": "xhigh"}
        fmt = payload["text"]["format"]
        assert payload["store"] is False
    else:
        assert payload["reasoning_effort"] == "xhigh"
        assert "store" not in payload
        fmt = payload["response_format"]["json_schema"]
    assert fmt["schema"] == floorplan_json_schema() and fmt["strict"] is True


def test_strict_schema_uses_supported_coordinate_arrays_and_nullable_fields():
    schema = floorplan_json_schema()
    wall = schema["$defs"]["Wall"]
    assert wall["additionalProperties"] is False
    assert set(wall["required"]) == {"start", "end", "thickness"}
    assert {"type": "null"} in wall["properties"]["thickness"]["anyOf"]
    coordinate = wall["properties"]["start"]
    assert coordinate["items"] == {"type": "number"}
    assert coordinate["minItems"] == coordinate["maxItems"] == 2
    encoded = json.dumps(schema)
    assert "prefixItems" not in encoded and "default" not in encoded
    assert "maxLength" not in encoded
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
def test_hosted_openings_survive_sdk_parsing_and_storage(endpoint, tmp_path, api):
    sample = dict(
        EXAMPLE,
        openings=[
            dict(wallIndex=0, offset=1.25, width=0.9, kind="door", hinge="end", swing="left")
        ],
    )
    endpoint["text"] = json.dumps(sample)
    store = Store(tmp_path / "openings.sqlite3")
    result = run(store, replace(config(endpoint), api_mode=api, response_format="json_schema"))
    assert result["success"] and result["warnings"] == []
    assert result["parsed"]["openings"] == sample["openings"]
    assert store.get(result["id"], "session-one")[0]["parsed"]["openings"] == sample["openings"]
    assert result["contract"] == "floorplan-v3" and result["renderer"] == "plotly-v2"


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
def test_refusals_remain_failed_with_reason_preserved(endpoint, tmp_path, api):
    endpoint["text"] = ""
    endpoint["refusal"] = "Cannot process this request."
    endpoint["output"] = [
        {
            "id": "msg-refusal",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "refusal", "refusal": endpoint["refusal"]}],
        }
    ]
    result = run(Store(tmp_path / "runs.sqlite3"), replace(config(endpoint), api_mode=api))
    assert not result["success"] and "refused" in result["error"]
    assert result["response"]["refusal"] == endpoint["refusal"]


@pytest.mark.parametrize(
    "status,detail",
    [
        ("incomplete", "max_output_tokens"),
        ("incomplete", "content_filter"),
        ("failed", None),
        ("cancelled", None),
        ("in_progress", None),
    ],
)
def test_unfinished_responses_remain_failed_even_with_valid_json(
    endpoint, tmp_path, status, detail
):
    endpoint["response_status"] = status
    endpoint["incomplete_details"] = {"reason": detail} if detail else None
    if status == "failed":
        endpoint["response_error"] = {"code": "server_error", "message": "DO NOT PERSIST"}
    store = Store(tmp_path / "runs.sqlite3")
    result = run(store, replace(config(endpoint), api_mode="responses"))
    assert result["raw"] == json.dumps(EXAMPLE) and not result["success"]
    assert result["parsed"] is None
    assert result["response"]["status"] == status
    assert "DO NOT PERSIST" not in json.dumps(result)
    if detail == "max_output_tokens":
        assert result["finish_reason"] == "length" and "budget exhausted" in result["error"]
    if status == "failed":
        assert result["response"]["error_code"] == "server_error"


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
@pytest.mark.parametrize("status", [401, 429, 500])
def test_http_errors_are_retained_without_retries_and_close_client(
    endpoint, tmp_path, monkeypatch, api, status
):
    sdk = client_module.AsyncOpenAI
    clients = []

    def track_client(**kwargs):
        instance = sdk(**kwargs)
        clients.append(instance)
        return instance

    monkeypatch.setattr(client_module, "AsyncOpenAI", track_client)
    endpoint["status"] = status
    store = Store(tmp_path / "runs.sqlite3")
    result = run(store, replace(config(endpoint), api_mode=api))
    assert not result["success"] and str(status) in result["error"]
    assert result["response"] == {"request_id": "req-local-test", "http_status": status}
    assert len(endpoint["requests"]) == 1 and clients[0].is_closed()
    assert "secret-test-key" not in json.dumps(result)
    assert "DO NOT EXPOSE" not in json.dumps(result)
    assert store.get(result["id"], "session-one")[0] == result


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"seed": 42}, "seed"),
        ({"token_parameter": "max_tokens"}, "max_output_tokens"),
        (
            {"api_mode": "chat_completions", "token_parameter": "max_output_tokens"},
            "max_output_tokens",
        ),
    ],
)
def test_api_parameter_conflicts_rejected_before_request(changes, match):
    settings = {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-6-astra",
        "prompt": DEFAULT_PROMPT,
    } | changes
    with pytest.raises(ValueError, match=match):
        RunConfig(**settings)


def test_explicit_model_settings_are_not_rewritten_from_model_name(endpoint):
    cfg = replace(
        config(endpoint),
        model="gpt-6-astra",
        temperature=0,
        token_parameter="max_tokens",
        reasoning_effort="none",
    )
    assert cfg.api == "chat_completions"
    assert cfg.parameters() == {"max_tokens": 32768, "temperature": 0, "reasoning_effort": "none"}
    assert (
        RunConfig("https://api.openai.com.example/v1", "model", DEFAULT_PROMPT).api
        == "chat_completions"
    )
    official = RunConfig(
        "https://api.openai.com/v1",
        "gpt-6-luna",
        DEFAULT_PROMPT,
        reasoning_effort="minimal",
        temperature=0,
    )
    assert official.parameters()["temperature"] == 0
    assert official.parameters()["reasoning"] == {"effort": "minimal"}
    chat = replace(official, api_mode="chat_completions")
    assert chat.parameters()["max_completion_tokens"] == 32768
    assert "max_output_tokens" not in chat.parameters()


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
def test_sdk_timeout_is_retained_and_transport_closed(endpoint, tmp_path, monkeypatch, api):
    sdk = client_module.AsyncOpenAI
    attempts = []

    async def timeout(request):
        attempts.append(request.url.path)
        raise httpx.ReadTimeout("DO NOT PERSIST", request=request)

    transport = httpx.AsyncClient(transport=httpx.MockTransport(timeout))
    monkeypatch.setattr(
        client_module, "AsyncOpenAI", lambda **kwargs: sdk(**kwargs, http_client=transport)
    )
    store = Store(tmp_path / "runs.sqlite3")
    result = run(store, replace(config(endpoint), api_mode=api))
    assert not result["success"] and "timed out" in result["error"]
    assert len(attempts) == 1 and transport.is_closed
    assert "DO NOT PERSIST" not in json.dumps(result)
    assert store.get(result["id"], "session-one")[0] == result


def test_responses_collects_text_after_reasoning_items(endpoint, tmp_path):
    raw = json.dumps(EXAMPLE)
    endpoint["output"] = [
        {"id": "reasoning-test", "type": "reasoning", "summary": []},
        {
            "id": "msg-test",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": raw[:30], "annotations": []},
                {"type": "output_text", "text": raw[30:], "annotations": []},
            ],
        },
    ]
    result = run(Store(tmp_path / "runs.sqlite3"), replace(config(endpoint), api_mode="responses"))
    assert result["raw"] == raw and result["success"]


def test_ui_recognition_history_and_export(endpoint, tmp_path):
    from floorplanbench.ui import build_app

    app = build_app(tmp_path / "ui.sqlite3")
    fn = next(fn.fn for fn in app.fns.values() if fn.fn and fn.fn.__name__ == "evaluate")
    callbacks = {fn.fn.__name__: fn.fn for fn in app.fns.values() if fn.fn}
    button = next(block for block in app.blocks.values() if block.elem_id == "run-button")
    heading = next(block for block in app.blocks.values() if block.elem_id == "output-heading")
    assert callbacks["prepare_recognition"]()[button]["interactive"] is False

    async def evaluate():
        return await fn(
            example_image(),
            endpoint["url"],
            "secret-test-key",
            "test-vlm",
            DEFAULT_PROMPT,
            32768,
            "auto",
            False,
            0,
            False,
            1,
            None,
            "xhigh",
            "json_schema",
            "high",
            2048,
            300,
            "ui-session",
            "responses",
        )

    updates = asyncio.run(evaluate())
    assert updates[button]["interactive"] is True
    from html import unescape

    assert unescape(updates[heading]) == "### test-vlm\n\nReconstruction"
    result = next(
        value
        for value in updates.values()
        if isinstance(value, dict) and "raw" in value and "api" in value
    )
    assert result["success"] and result["api"] == "responses"
    assert result["request_parameters"]["reasoning"] == {"effort": "xhigh"}
    assert endpoint["requests"][0]["path"] == "/v1/responses"

    restored = callbacks["restore_result"](result["id"], "ui-session")
    assert restored[heading] == updates[heading]
    history = next(
        value for value in restored.values() if isinstance(value, dict) and "choices" in value
    )
    assert history["choices"][0][1] == result["id"]
    assert "Reconstructed" in history["choices"][0][0]
    exported = callbacks["export_result"](result, "ui-session")
    from pathlib import Path

    path = Path(exported["value"])
    assert json.loads(path.read_text()) == result
    path.unlink()


def test_upload_history_preserves_failures_without_accuracy_scores(endpoint, tmp_path):
    store = Store(tmp_path / "runs.sqlite3")
    good = run(store, config(endpoint))
    endpoint["text"] = "I cannot read this plan."
    bad = run(store, config(endpoint))
    assert not bad["success"] and bad["raw"] == endpoint["text"]
    assert len(store.history("session-one")) == 2
    assert store.get(bad["id"], "session-one")[0] == bad
    with pytest.raises(ValueError):
        store.get(good["id"], "different-session")


def test_empty_gradio_fields_support_no_auth_local_services(endpoint, tmp_path):
    cfg = replace(config(endpoint), scale_hint=None)
    result = asyncio.run(
        run_model(Store(tmp_path / "runs.sqlite3"), "session", cfg, None, example_image())
    )
    assert result["success"]
    assert endpoint["requests"][0]["auth"] == "Bearer not-required"
    with pytest.raises(ValueError):
        replace(cfg, model=None)


def test_protocol_isolates_conditions_but_compares_models_and_endpoints(endpoint):
    cfg = config(endpoint)
    image = prepare_image(example_image())
    original = cfg.protocol(image)
    assert (
        replace(cfg, model="another-vlm", base_url="https://example.org/v1").protocol(image)
        == original
    )
    for variant in [
        replace(cfg, prompt=DEFAULT_PROMPT + "\nTest"),
        replace(cfg, temperature=0.2),
        replace(cfg, scale_hint="10m"),
        replace(cfg, image_detail="high"),
        replace(cfg, max_tokens=3000),
        replace(cfg, response_format="json_object"),
        replace(cfg, api_mode="responses"),
    ]:
        assert variant.protocol(image) != original
    assert cfg.protocol(image + b"changed") != original


@pytest.mark.parametrize(
    "value",
    [
        "file:///tmp/key",
        "https://user:secret@example.com/v1",
        "https://example.com/v1?api_key=secret",
        "example.com",
        "",
    ],
)
def test_invalid_endpoints(value):
    with pytest.raises(ValueError):
        endpoint_url(value)


def test_image_preprocessing_composites_alpha_and_keeps_aspect_ratio():
    image = Image.new("RGBA", (4000, 2000), (0, 0, 0, 0))
    result = Image.open(io.BytesIO(prepare_image(image, 1024)))
    assert result.size == (1024, 512) and result.getpixel((0, 0)) == (255, 255, 255)


def test_parser_keeps_failures_and_reports_fence_tolerance():
    plan, notes = parse_plan("```json\n" + json.dumps(EXAMPLE) + "\n```")
    assert renderable(plan) and notes
    for changed in [
        dict(EXAMPLE, confidence=2),
        dict(EXAMPLE, confidence=True),
        dict(EXAMPLE, confidence=float("nan")),
        dict(EXAMPLE, walls="not an array"),
    ]:
        with pytest.raises(ValueError):
            parse_plan(json.dumps(changed))
    empty, _ = parse_plan(json.dumps(dict(EXAMPLE, walls=[], rooms=[])))
    assert not renderable(empty)


def test_out_of_schema_fields_are_explicitly_ignored():
    plan, warnings = parse_plan(json.dumps(dict(EXAMPLE, doors=[{"invented": True}])))
    assert "doors" in warnings[0]
    assert "doors" not in plan.model_dump()


def test_truncated_output_is_recorded_with_warning(endpoint, tmp_path):
    endpoint["finish"] = "length"
    endpoint["text"] = '{"walls": ['
    result = run(Store(tmp_path / "runs.sqlite3"), config(endpoint))
    assert not result["success"] and "length" in result["warnings"][0]
    assert "truncated" in result["error"]


def test_reasoning_exhausted_budget_is_not_reported_as_invalid_json(endpoint, tmp_path):
    endpoint["finish"] = "length"
    endpoint["text"] = ""
    result = run(Store(tmp_path / "runs.sqlite3"), replace(config(endpoint), max_tokens=2000))
    assert not result["success"]
    assert "budget exhausted" in result["error"] and "2000" in result["error"]
    assert "JSON" not in result["error"]
    assert not result["warnings"]


def test_renderer_escapes_model_labels_and_uses_metric_wall_thickness():
    sample = json.loads(json.dumps(EXAMPLE))
    sample["rooms"][0]["name"] = '<script>alert("x")</script>'
    plan, _ = parse_plan(json.dumps(sample))
    figure = plot_plan(plan)
    assert "<script>" not in figure.to_json()
    assert figure.layout.shapes[0].type == "path"
    assert "0.08" in figure.layout.shapes[0].path


def test_shapely_checks_invalid_polygons_without_repair_and_places_labels_inside():
    sample = json.loads(json.dumps(EXAMPLE))
    sample["walls"] = []
    sample["rooms"] = [{"name": "crossing", "polygon": [[0, 0], [2, 2], [0, 2], [2, 0]]}]
    plan, notes = parse_plan(json.dumps(sample))
    assert not renderable(plan) and notes
    assert plan.rooms[0].polygon == [(0, 0), (2, 2), (0, 2), (2, 0)]
    sample["rooms"][0]["polygon"] = [[0, 0], [4, 0], [4, 1], [1, 1], [1, 4], [0, 4]]
    plan, _ = parse_plan(json.dumps(sample))
    from shapely.geometry import box

    label = plot_plan(plan).layout.images[0]
    bounds = box(
        label.x - label.sizex / 2,
        label.y - label.sizey / 2,
        label.x + label.sizex / 2,
        label.y + label.sizey / 2,
    )
    assert room_polygon(plan.rooms[0]).contains(bounds)
