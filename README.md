# FloorPlanBench

A single-process Gradio app for comparing VLM floor-plan recognition and reconstruction through OpenAI-compatible APIs.

## Run

```bash
pip install -r requirements.txt
python app.py
```

Requires Python 3.11+. Open <http://127.0.0.1:7860>.
Alternatively, use `uv sync --locked` followed by `uv run python app.py`.

## Recognize a floor plan

Enter a Base URL, API Key, and Model name. Upload a floor plan and click **Recognize**.
The page starts empty. Advanced settings and the recognition prompt are collapsed by default.

| Setting | OpenAI endpoint (Auto) | Other compatible endpoints (Auto) |
| --- | --- | --- |
| API type | Responses | Chat Completions |
| Token parameter | `max_output_tokens` | `max_tokens`; can be changed to `max_completion_tokens` |
| Response format | Strict `json_schema` | Prompt-only output, without a format parameter |
| Temperature / top_p | Omitted by default | Omitted by default; can be enabled |

Auto recognizes `api.openai.com` and its regional subdomains. Proxy URLs are treated as third-party providers.
Either API can be selected explicitly. There is no automatic API switching or request retry.
Providers must support image inputs for the selected API; local services without authentication can leave API Key empty.

The default output budget is **16,384 tokens**, which may include reasoning tokens.
Increase Max output tokens or reduce Reasoning effort if a response exhausts its budget.
`default` uses the model's own setting; supported reasoning and sampling options vary by provider.
The provider validates model-specific parameter support.

Response format supports `json_schema`, `json_object`, and `default`.
Strict schemas use two-number coordinate arrays and required nullable fields.
With Responses, use `auto` or `max_output_tokens` and leave Seed empty.
OpenAI Chat Completions uses `max_completion_tokens` in Auto mode.
See the [Responses migration guide](https://developers.openai.com/api/docs/guides/migrate-to-responses),
[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs), and
[model parameter reference](https://developers.openai.com/api/docs/guides/latest-model).

Plotly reconstructs walls and rooms beside the source image, with pan and zoom.
Structured JSON, raw responses, run metadata, and session history are under **Result details**.
Bilingual room names use their English label in the drawing; original labels remain in hover text and JSON.

The default prompt comes from [Pascal Editor](https://github.com/pascalorg/editor/blob/559c0d16133b3aa82fa73b5e7ba42d49aaf160d4/packages/mcp/src/tools/vision/analyze-floorplan-image.ts).
Its contract uses wall coordinates in metres and room polygons, without openings or furniture.
The original prompt, source reference, and MIT license are in `floorplanbench/prompts/`.

## Run history and exports

Results can be exported as JSON. SQLite stores runs in `data/bench.sqlite3`; API keys are not persisted.
Images are sent to the configured provider. Uploaded images do not receive reference-based accuracy scores.

Run metadata retains actual request parameters and provider-supplied IDs, model names, fingerprints,
refusal reasons, and usage details when available. Refused, failed, or incomplete
responses count as failures; raw responses are retained. HTTP failures retain status codes and request IDs,
without full provider error bodies. Seed does not guarantee determinism, and default reasoning effort
does not establish equal reasoning budgets across models.

OpenAI requests and all Responses requests send `store=false` without creating a server-side conversation.
This does not enable Zero Data Retention. Third-party Chat Completions omits `store` for compatibility.
See [OpenAI data controls](https://developers.openai.com/api/docs/guides/your-data).

## Development

```bash
uv sync --locked --group dev
uv run pytest
uv run ruff check .
```

The app reuses Gradio, the OpenAI Python SDK, Plotly, Shapely, Pydantic, and SQLite.
Tests use local mock endpoints without paid model calls. Synthetic inputs are in `tests/samples.py`.

Dependencies are defined in `pyproject.toml` and `uv.lock`. Regenerate pip requirements with:

```bash
uv export --locked --no-dev --no-hashes --no-emit-project --format requirements-txt -o requirements.txt
```
