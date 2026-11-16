import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime

from openai import APIConnectionError, APIStatusError, APITimeoutError
from PIL import Image

from .client import RunConfig, digest, prepare_image, recognize
from .schema import CONTRACT_VERSION, RENDERER_VERSION, parse_plan, renderable
from .storage import Store


def safe_error(exc: Exception) -> str:
    if isinstance(exc, APITimeoutError):
        return "Request timed out. Increase the timeout in Advanced settings."
    if isinstance(exc, APIConnectionError):
        return "Connection failed. Check the Base URL and network."
    if isinstance(exc, APIStatusError):
        return f"Model service returned HTTP {exc.status_code}."
    # Provider exception strings may contain Authorization headers or request data.
    return f"Request failed ({type(exc).__name__})."


async def run_model(
    store: Store,
    session: str,
    config: RunConfig,
    api_key: str,
    image: Image.Image,
):
    image_bytes = prepare_image(image, config.max_side)
    run = {
        "id": uuid.uuid4().hex,
        "created": datetime.now(UTC).isoformat(),
        "model": config.model,
        "endpoint": config.base_url,
        "protocol": config.protocol(image_bytes),
        "contract": CONTRACT_VERSION,
        "renderer": RENDERER_VERSION,
        "image_sha256": digest(image_bytes),
        "prompt_sha256": digest(config.prompt),
        "config": asdict(config),
        "success": False,
        "raw": "",
        "parsed": None,
        "warnings": [],
        "error": None,
        "usage": {},
        "finish_reason": None,
        "api": config.api,
        "request_parameters": config.parameters(),
        "response": {},
    }
    start = time.perf_counter()
    try:
        run["raw"], run["usage"], run["finish_reason"] = await recognize(
            config, api_key, image_bytes, metadata=run["response"]
        )
    except Exception as exc:
        run["error"] = safe_error(exc)
        if isinstance(exc, APIStatusError):
            run["response"].update(request_id=exc.request_id, http_status=exc.status_code)
    run["seconds"] = round(time.perf_counter() - start, 4)
    if run["error"] is None:
        if run["response"].get("refusal"):
            run["error"] = "The model refused the request. See Run metadata for the reason."
        elif config.api == "responses" and run["response"].get("status") != "completed":
            run["error"] = (
                f"Output budget exhausted ({config.max_tokens} tokens); response incomplete."
                if run["finish_reason"] == "length"
                else f"Model response incomplete ({run['finish_reason']}). See Run metadata."
            )
        elif not run["raw"].strip():
            if run["finish_reason"] == "length":
                run["error"] = (
                    f"Output budget exhausted ({config.max_tokens} tokens); no content returned. "
                    "Increase Max output tokens or reduce Reasoning effort in Advanced settings."
                )
            else:
                run["error"] = (
                    f"The model returned no content (finish reason: {run['finish_reason']})."
                )
        else:
            try:
                plan, run["warnings"] = parse_plan(run["raw"])
                run["parsed"] = plan.model_dump(mode="json")
                run["success"] = renderable(plan)
                if not run["success"]:
                    run["error"] = "JSON parsed, but contains no renderable walls or room polygons."
            except ValueError:
                run["error"] = (
                    "JSON truncated at the token limit. Increase Max output tokens in Advanced settings."
                    if run["finish_reason"] == "length"
                    else "Response does not match the selected JSON contract. See Raw response."
                )
        if run["raw"].strip() and run["finish_reason"] not in {"stop", "end_turn", "completed"}:
            run["warnings"].append(f"Finish reason: {run['finish_reason']}")
    store.save(run, session, image_bytes)
    return run
