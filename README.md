---
title: FloorPlanBench
emoji: 🏠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: space.py
short_description: Compare VLM floor-plan recognition and reconstruction
license: apache-2.0
tags:
  - floor-plan
  - vision-language-model
  - benchmark
---

# FloorPlanBench

A single-process Gradio app for comparing VLM floor-plan recognition and reconstruction through OpenAI-compatible APIs.

## Run

```bash
pip install -r requirements.txt
python app.py
```

Requires Python 3.11+. Open <http://127.0.0.1:7860>.
Alternatively, use `uv sync --locked` followed by `uv run python app.py`.

## Hugging Face Space

The hosted app is [thomas-yanxin/FloorPlanBench](https://huggingface.co/spaces/thomas-yanxin/FloorPlanBench).
Visitors supply their own model endpoint and API key. The provider must be reachable
from the Space; a visitor's `localhost` endpoint is not accessible from Hugging Face.
History uses the Space's temporary disk and can be lost when the Space restarts.
Export results you want to keep.

The Space uses the free ZeroGPU tier. `space.py` registers the platform-required GPU
function without calling it; recognition and rendering do not reserve a GPU.
The regular local entry point remains `app.py`.

To sync the checkout again, install the `hf` CLI, run `hf auth login`, then:

```bash
uv run python scripts/sync_huggingface.py
```

This uploads Git-tracked files and the deployment entry points, generating the
Space's requirements from `uv.lock` while leaving Gradio, Hugging Face Hub, and
Gradio's Pydantic dependency under platform management. Local data, credentials,
caches, and Git history are not uploaded. Uploads rebuild and restart the Space.

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

The default output budget is **32,768 tokens**, which may include reasoning tokens.
Increase Max output tokens or reduce Reasoning effort if a response exhausts its budget.
`default` uses the model's own setting; supported reasoning and sampling options vary by provider.
The default request timeout is **300 seconds**. The provider validates model-specific parameter support.

Response format supports `json_schema`, `json_object`, and `default`.
Strict schemas use two-number coordinate arrays and required nullable fields.
With Responses, use `auto` or `max_output_tokens` and leave Seed empty.
OpenAI Chat Completions uses `max_completion_tokens` in Auto mode.
See the [Responses migration guide](https://developers.openai.com/api/docs/guides/migrate-to-responses),
[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs), and
[model parameter reference](https://developers.openai.com/api/docs/guides/latest-model).

Plotly reconstructs walls and rooms beside the source image, with pan and zoom.
Wall faces are united into a continuous metric surface with mitred junctions; free ends and
unclassified gaps are preserved. Labels fit inside room polygons and scale with the drawing,
including room areas when space permits. Rendering does not snap or correct model coordinates.
Structured JSON, raw responses, run metadata, and session history are under **Result details**.
Bilingual room names use their English label in the drawing; original labels remain in hover text and JSON.

The default prompt is `floorplanbench/prompts/reconstruction_floorplan.txt`. It specifies wall
centrelines, interior room boundaries, consistent junctions and optional doors, windows and passages.
Each opening references a wall by zero-based `wallIndex`; `offset` and `width` are distances in metres
along that wall from its start. The renderer cuts the host wall and draws jambs, window lines, and
door leaves/arcs when the model supplies a hinge and swing direction. Missing swings are not guessed.
Out-of-bounds openings are skipped with a warning. Furniture is outside this contract.

This extends the wall/room contract from
[Pascal Editor](https://github.com/pascalorg/editor/blob/559c0d16133b3aa82fa73b5e7ba42d49aaf160d4/packages/mcp/src/tools/vision/analyze-floorplan-image.ts).
The unchanged original prompt, source reference, and MIT license remain in `floorplanbench/prompts/`.
Old saved results without openings still render; their unclassified gaps cannot recover missing door
or window information. Use the new default prompt for future recognition runs. Changing prompts or
the output contract changes benchmark conditions; new runs record contract and renderer versions.

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
