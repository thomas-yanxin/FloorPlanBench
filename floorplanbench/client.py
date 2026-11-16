"""Provider calls go exclusively through the official OpenAI Python SDK."""

import base64
import hashlib
import io
import json
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from openai import AsyncOpenAI
from PIL import Image, ImageOps

from .schema import CONTRACT_VERSION, RENDERER_VERSION, floorplan_json_schema


def digest(value: bytes | str) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def endpoint_url(value: str) -> str:
    parts = urlsplit((value or "").strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(
            "Base URL must be a complete HTTP(S) URL, such as https://api.openai.com/v1."
        )
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError(
            "Base URL cannot include credentials, query parameters, or fragments. Enter credentials in API Key."
        )
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def prepare_image(image: Image.Image, max_side: int = 2048) -> bytes:
    if image is None:
        raise ValueError("Upload a floor plan first.")
    if image.width * image.height > 40_000_000:
        raise ValueError("Image exceeds 40 megapixels. Resize it before uploading.")
    image = ImageOps.exif_transpose(image).convert("RGBA")
    background = Image.new("RGBA", image.size, "white")
    background.alpha_composite(image)
    image = background.convert("RGB")
    if max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    if stream.tell() > 20 * 1024 * 1024:
        raise ValueError("Encoded image exceeds 20 MB. Reduce the image resolution.")
    return stream.getvalue()


@dataclass(frozen=True)
class RunConfig:
    base_url: str
    model: str
    prompt: str
    scale_hint: str = ""
    max_tokens: int = 16384
    token_parameter: str = "auto"
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    response_format: str = "auto"
    reasoning_effort: str = "default"
    image_detail: str = "auto"
    max_side: int = 2048
    timeout: float = 120
    api_mode: str = "auto"

    def __post_init__(self):
        object.__setattr__(self, "base_url", endpoint_url(self.base_url))
        object.__setattr__(self, "model", (self.model or "").strip())
        object.__setattr__(self, "prompt", self.prompt or "")
        object.__setattr__(self, "scale_hint", self.scale_hint or "")
        if not self.model or not self.prompt.strip():
            raise ValueError("Enter a Model name and Recognition prompt.")
        if self.api_mode not in {"auto", "responses", "chat_completions"}:
            raise ValueError("Unsupported API type.")
        if self.token_parameter not in {
            "auto",
            "max_tokens",
            "max_completion_tokens",
            "max_output_tokens",
        }:
            raise ValueError("Unsupported token parameter.")
        if self.image_detail not in {"auto", "low", "high"}:
            raise ValueError("Unsupported image detail.")
        if not 1 <= self.max_tokens <= 65536 or not 1 <= self.timeout <= 600:
            raise ValueError("Token limit or timeout is out of range.")
        if self.max_side not in {0, 1024, 2048, 4096}:
            raise ValueError("Unsupported image resolution.")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2.")
        if self.top_p is not None and not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1.")
        if self.response_format not in {"auto", "default", "json_object", "json_schema"}:
            raise ValueError("Unsupported response_format.")
        if self.reasoning_effort not in {
            "default",
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ValueError("Unsupported reasoning_effort.")
        self.parameters()  # Reject API-level conflicts before creating a benchmark attempt.

    @property
    def is_openai(self) -> bool:
        host = urlsplit(self.base_url).hostname
        return host == "api.openai.com" or host.endswith(".api.openai.com")

    @property
    def api(self) -> str:
        if self.api_mode == "auto":
            return "responses" if self.is_openai else "chat_completions"
        return self.api_mode

    @property
    def output_format(self) -> str:
        if self.response_format == "auto":
            return "json_schema" if self.is_openai else "default"
        return self.response_format

    def parameters(self) -> dict:
        token = self.token_parameter
        if self.api == "responses":
            if token not in {"auto", "max_output_tokens"}:
                raise ValueError("Use auto or max_output_tokens with Responses.")
            if self.seed is not None:
                raise ValueError(
                    "Responses does not support seed. Leave it empty or select Chat Completions."
                )
            token = "max_output_tokens"
        else:
            if token == "max_output_tokens":
                raise ValueError("Chat Completions does not support max_output_tokens.")
            if token == "auto":
                token = "max_completion_tokens" if self.is_openai else "max_tokens"
        params = {token: self.max_tokens}
        for name in ("temperature", "top_p", "seed"):
            value = getattr(self, name)
            if value is not None:
                params[name] = value
        if self.output_format != "default":
            fmt = {"type": self.output_format}
            if self.output_format == "json_schema":
                definition = {
                    "name": "floor_plan",
                    "strict": True,
                    "schema": floorplan_json_schema(),
                }
                if self.api == "responses":
                    fmt.update(definition)
                else:
                    fmt["json_schema"] = definition
            if self.api == "responses":
                params["text"] = {"format": fmt}
            else:
                params["response_format"] = fmt
        if self.reasoning_effort != "default":
            if self.api == "responses":
                params["reasoning"] = {"effort": self.reasoning_effort}
            else:
                params["reasoning_effort"] = self.reasoning_effort
        if self.is_openai or self.api == "responses":
            params["store"] = False
        return params

    def protocol(self, image_bytes: bytes) -> str:
        # Model/provider deliberately excluded, so identical tasks can be compared across them.
        return digest(
            json.dumps(
                {
                    "image": digest(image_bytes),
                    "prompt": self.prompt,
                    "scale_hint": self.scale_hint,
                    "parameters": self.parameters(),
                    "api": self.api,
                    "image_detail": self.image_detail,
                    "contract": CONTRACT_VERSION,
                    "renderer": RENDERER_VERSION,
                },
                sort_keys=True,
                ensure_ascii=False,
            )
        )


async def recognize(
    config: RunConfig, api_key: str, image_bytes: bytes, *, metadata: dict
) -> tuple[str, dict, str]:
    instruction = (
        f"Analyze this floor plan. Scale hint: {config.scale_hint}. "
        "Return ONLY the JSON described by the system prompt."
        if config.scale_hint.strip()
        else "Analyze this floor plan. Return ONLY the JSON described by the system prompt."
    )
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode()
    # No fallback model, no hidden retry/repair calls, no server-wide API-key fallback.
    async with AsyncOpenAI(
        base_url=config.base_url,
        api_key=(api_key or "").strip() or "not-required",
        timeout=config.timeout,
        max_retries=0,
    ) as client:
        if config.api == "responses":
            response = await client.responses.create(
                model=config.model,
                input=[
                    {"role": "system", "content": config.prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": data_url,
                                "detail": config.image_detail,
                            },
                            {"type": "input_text", "text": instruction},
                        ],
                    },
                ],
                **config.parameters(),
            )
            metadata.update(
                response_id=response.id,
                request_id=response._request_id,
                model=response.model,
                status=response.status,
                incomplete_details=(
                    response.incomplete_details.model_dump()
                    if response.incomplete_details
                    else None
                ),
                error_code=response.error.code if response.error else None,
                reasoning=response.reasoning.model_dump() if response.reasoning else None,
                service_tier=response.service_tier,
            )
            refusals = [
                part.refusal
                for item in response.output
                if item.type == "message"
                for part in item.content
                if part.type == "refusal"
            ]
            metadata["refusal"] = "\n".join(refusals) or None
            reason = response.status or "unknown"
            if response.incomplete_details:
                reason = response.incomplete_details.reason or reason
            if reason == "max_output_tokens":
                reason = "length"
            return (
                response.output_text,
                response.usage.model_dump() if response.usage else {},
                reason,
            )
        response = await client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": config.prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                                "detail": config.image_detail,
                            },
                        },
                        {"type": "text", "text": instruction},
                    ],
                },
            ],
            **config.parameters(),
        )
    metadata.update(
        response_id=response.id,
        request_id=response._request_id,
        model=response.model,
        system_fingerprint=response.system_fingerprint,
        service_tier=response.service_tier,
    )
    if not response.choices:
        return "", response.usage.model_dump() if response.usage else {}, "no_choices"
    choice = response.choices[0]
    metadata["refusal"] = choice.message.refusal
    return (
        choice.message.content or "",
        response.usage.model_dump() if response.usage else {},
        choice.finish_reason or "unknown",
    )
