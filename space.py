"""ZeroGPU entry point; model requests continue to run through the configured API."""

import os
import runpy

import spaces


@spaces.GPU(duration=1)
def _unused_gpu() -> None:
    """Register ZeroGPU support without reserving a GPU for API requests."""


if __name__ == "__main__":
    os.environ.setdefault("GRADIO_SERVER_NAME", "0.0.0.0")
    runpy.run_path("app.py", run_name="__main__")
