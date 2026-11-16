import os

from floorplanbench.ui import LOCALE_MODULE, ROOT, build_app, theme

demo = build_app()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4, max_size=32).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7860")),
        share=False,
        theme=theme(),
        css=(ROOT / "style.css").read_text(),
        # Set Gradio's locale before its upload controls and menus are mounted.
        head=(
            '<script type="module">'
            f'import {{ changeLocale }} from "./assets/{LOCALE_MODULE}";'
            'changeLocale("en");'
            "</script>"
        ),
        max_file_size="20mb",
        show_error=False,
        blocked_paths=[str(ROOT.parent / "data"), str(ROOT.parent / ".env")],
    )
