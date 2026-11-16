"""Controls, uploads and event handling use native Gradio components."""

import io
import json
import os
import tempfile
import uuid
from pathlib import Path

import gradio as gr
from PIL import Image

from .client import RunConfig
from .render import plot_plan
from .schema import DEFAULT_PROMPT, FloorPlan
from .service import run_model
from .storage import Store

ROOT = Path(__file__).parent
LOCALE_MODULE = next(
    (Path(gr.__file__).parent / "templates/frontend/assets").glob("core-*.js")
).name


def theme():
    return gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=["Inter", "PingFang SC", "Microsoft YaHei", "sans-serif"],
    ).set(
        body_background_fill="white",
        body_text_color="#263346",
        block_background_fill="white",
        block_border_color="#e1e7ef",
        block_radius="8px",
        block_label_text_size="13px",
        input_background_fill="white",
        input_border_color="#d4dde9",
        input_radius="6px",
        button_large_text_size="14px",
        button_small_text_size="13px",
        body_text_size="14px",
        section_header_text_size="14px",
    )


def build_app(db_path: str | Path | None = None) -> gr.Blocks:
    store = Store(db_path or os.environ.get("FLOORPLANBENCH_DB", "data/bench.sqlite3"))
    downloads = Path(tempfile.mkdtemp(prefix="floorplanbench-exports-"))

    def history_choices(session):
        choices = []
        for run in store.history(session):
            created = run["created"][:19].replace("T", " ")
            state = "Reconstructed" if run["success"] else "Failed"
            choices.append((f"{run['model']} · {created} · {state} · {run['id'][:8]}", run["id"]))
        return choices

    def result_values(run, session):
        plan = FloorPlan.model_validate_json(json.dumps(run["parsed"])) if run["parsed"] else None
        if run["success"]:
            usage = run["usage"].get("total_tokens")
            note = (
                f"Reconstructed · {len(plan.walls)} walls · {len(plan.rooms)} rooms · "
                f"{run['seconds']:.2f} s"
            )
            if usage is not None:
                note += f" · {usage} tokens"
        else:
            note = run["error"]
        return {
            reconstruction: plot_plan(plan) if run["success"] else None,
            status: gr.update(value=note, visible=True),
            structured: run["parsed"],
            raw: run["raw"],
            metadata: {k: v for k, v in run.items() if k not in {"raw", "parsed"}},
            restore: gr.update(choices=history_choices(session), value=None, interactive=True),
            run_file: gr.update(value=None, visible=False),
            current: run,
            original: gr.update(interactive=True),
            run_button: gr.update(interactive=True),
        }

    with gr.Blocks(title="FloorPlanBench", analytics_enabled=False) as demo:
        session = gr.State(lambda: uuid.uuid4().hex)
        current = gr.State(None)
        gr.HTML(
            """<div class="brandline"><svg viewBox="0 0 32 32" fill="none" aria-hidden="true">
            <path d="M3 3h26v26H19v-9H13v9H3V3Z M3 13h9V3 M21 3v10h8"
            stroke="#263346" stroke-width="2.3" stroke-linejoin="miter"/></svg>
            <strong>FloorPlanBench</strong></div>""",
            elem_id="brand",
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=270, elem_id="settings"):
                gr.Markdown("## Model connection")
                base_url = gr.Textbox(label="Base URL", value="https://api.openai.com/v1")
                api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    html_attributes={"autocomplete": "off"},
                )
                model = gr.Textbox(label="Model name")
                with gr.Accordion("Advanced settings", open=False):
                    api_mode = gr.Dropdown(
                        [
                            ("Auto", "auto"),
                            ("Responses", "responses"),
                            ("Chat Completions", "chat_completions"),
                        ],
                        value="auto",
                        label="API type",
                    )
                    max_tokens = gr.Number(
                        value=16384,
                        minimum=256,
                        maximum=65536,
                        precision=0,
                        label="Max output tokens",
                    )
                    token_parameter = gr.Dropdown(
                        ["auto", "max_tokens", "max_completion_tokens", "max_output_tokens"],
                        value="auto",
                        label="Token parameter",
                    )
                    send_temp = gr.Checkbox(value=False, label="Send temperature")
                    temperature = gr.Slider(0, 2, value=0, step=0.1, label="Temperature")
                    send_top_p = gr.Checkbox(value=False, label="Send top_p")
                    top_p = gr.Slider(0, 1, value=1, step=0.05, label="Top p")
                    seed = gr.Number(value=None, precision=0, label="Seed")
                    reasoning = gr.Dropdown(
                        ["default", "none", "minimal", "low", "medium", "high", "xhigh", "max"],
                        value="default",
                        label="Reasoning effort",
                    )
                    response_format = gr.Dropdown(
                        ["auto", "default", "json_object", "json_schema"],
                        value="auto",
                        label="Response format",
                    )
                    detail = gr.Dropdown(
                        ["auto", "low", "high"], value="auto", label="Image detail"
                    )
                    max_side = gr.Dropdown(
                        [
                            ("1024 px", 1024),
                            ("2048 px", 2048),
                            ("4096 px", 4096),
                            ("Original resolution", 0),
                        ],
                        value=2048,
                        label="Max image dimension",
                    )
                    timeout = gr.Slider(10, 300, value=120, step=10, label="Timeout / s")
                with gr.Accordion("Recognition prompt", open=False):
                    prompt = gr.Textbox(value=DEFAULT_PROMPT, show_label=False, lines=15)
                    reset_prompt = gr.Button("Reset to default", size="sm")
                run_button = gr.Button(
                    "Recognize", variant="primary", elem_id="run-button", interactive=False
                )
            with gr.Column(scale=4, min_width=340, elem_id="workspace"):
                with gr.Row(equal_height=False):
                    with gr.Column(min_width=280, elem_classes="canvas-column"):
                        gr.Markdown("### Original floor plan", elem_classes="canvas-heading")
                        original = gr.Image(
                            value=None,
                            type="pil",
                            image_mode="RGBA",
                            sources=["upload", "clipboard"],
                            label="Upload floor plan",
                            show_label=False,
                            height=420,
                            format="png",
                            elem_classes="canvas",
                            buttons=["download", "fullscreen"],
                        )
                    with gr.Column(min_width=280, elem_classes="canvas-column"):
                        gr.Markdown("### Reconstruction", elem_classes="canvas-heading")
                        reconstruction = gr.Plot(
                            value=None,
                            show_label=False,
                            elem_classes="canvas",
                            elem_id="reconstruction",
                        )
                status = gr.Markdown("", elem_id="status", visible=False)
                with gr.Accordion("Result details", open=False):
                    with gr.Tabs():
                        with gr.Tab("Structured JSON"):
                            structured = gr.JSON(
                                value=None,
                                show_label=False,
                                open=True,
                            )
                        with gr.Tab("Raw response"):
                            raw = gr.Textbox(
                                label="Raw response", show_label=False, lines=10, interactive=False
                            )
                            metadata = gr.JSON(label="Run metadata", open=False)
                        with gr.Tab("History"):
                            restore = gr.Dropdown(choices=[], label="Session runs")
                    export_run = gr.Button("Export result JSON", size="sm")
                    run_file = gr.File(label="Result file", interactive=False, visible=False)
        outputs = [
            reconstruction,
            status,
            structured,
            raw,
            metadata,
            restore,
            run_file,
            current,
            original,
            run_button,
        ]

        def clear_result():
            return {
                reconstruction: None,
                status: gr.update(value="", visible=False),
                structured: None,
                raw: "",
                metadata: None,
                restore: gr.update(interactive=True),
                run_file: gr.update(value=None, visible=False),
                current: None,
                original: gr.update(interactive=True),
                run_button: gr.update(interactive=True),
            }

        async def evaluate(
            image,
            endpoint,
            key,
            model_name,
            system_prompt,
            token_limit,
            token_param,
            use_temp,
            temp,
            use_top,
            top,
            seed_value,
            effort,
            fmt,
            image_detail,
            side,
            request_timeout,
            sid,
            selected_api,
        ):
            running = clear_result()
            running.update(
                {
                    status: gr.update(value="Recognizing…", visible=True),
                    original: gr.update(interactive=False),
                    restore: gr.update(interactive=False),
                    run_button: gr.update(interactive=False),
                }
            )
            yield running
            try:
                config = RunConfig(
                    base_url=endpoint,
                    model=model_name,
                    prompt=system_prompt,
                    max_tokens=int(token_limit),
                    token_parameter=token_param,
                    temperature=temp if use_temp else None,
                    top_p=top if use_top else None,
                    seed=int(seed_value) if seed_value is not None else None,
                    reasoning_effort=effort,
                    response_format=fmt,
                    image_detail=image_detail,
                    max_side=int(side),
                    timeout=request_timeout,
                    api_mode=selected_api,
                )
                result = await run_model(store, sid, config, key, image)
            except ValueError as exc:
                cleared = clear_result()
                cleared[status] = gr.update(value=str(exc), visible=True)
                yield cleared
                return
            yield result_values(result, sid)

        run_button.click(
            evaluate,
            inputs=[
                original,
                base_url,
                api_key,
                model,
                prompt,
                max_tokens,
                token_parameter,
                send_temp,
                temperature,
                send_top_p,
                top_p,
                seed,
                reasoning,
                response_format,
                detail,
                max_side,
                timeout,
                session,
                api_mode,
            ],
            outputs=outputs,
            concurrency_limit=4,
            trigger_mode="once",
            api_name=False,
            show_progress="hidden",
        )

        def image_changed(image):
            values = clear_result()
            values[run_button] = gr.update(interactive=image is not None)
            return values

        original.input(
            image_changed, inputs=original, outputs=outputs, queue=False, show_progress="hidden"
        )

        reset_prompt.click(
            lambda: DEFAULT_PROMPT, outputs=prompt, queue=False, show_progress="hidden"
        )

        def restore_result(run_id, sid):
            if not run_id:
                return {component: gr.skip() for component in outputs}
            try:
                result, image_bytes = store.get(run_id, sid)
            except ValueError as exc:
                raise gr.Error(str(exc)) from None
            values = result_values(result, sid)
            values[original] = gr.update(
                value=Image.open(io.BytesIO(image_bytes)).copy(), interactive=True
            )
            return values

        restore.input(
            restore_result, inputs=[restore, session], outputs=outputs, show_progress="hidden"
        )

        def export_result(run, sid):
            if not run:
                raise gr.Error("No result yet.")
            result, _ = store.get(run["id"], sid)
            path = downloads / f"{result['id']}.json"
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            return gr.update(value=str(path), visible=True)

        export_run.click(
            export_result,
            inputs=[current, session],
            outputs=run_file,
            api_name=False,
            show_progress="hidden",
        )
    return demo
