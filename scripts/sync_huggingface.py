"""Upload this checkout to its Space without local data or platform-managed pins."""

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "thomas-yanxin/FloorPlanBench"


def main() -> None:
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    paths = {path for path in tracked if path} | {"space.py", "scripts/sync_huggingface.py"}
    with tempfile.TemporaryDirectory(prefix="floorplanbench-space-") as temporary:
        staging = Path(temporary)
        for name in sorted(paths):
            source = ROOT / name
            if source.is_file():
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        subprocess.run(
            [
                "uv",
                "export",
                "--quiet",
                "--locked",
                "--no-dev",
                "--no-hashes",
                "--no-header",
                "--no-emit-project",
                "--prune",
                "gradio",
                # Spaces installs gradio[mcp], which constrains Pydantic separately.
                "--no-emit-package",
                "pydantic",
                "--no-emit-package",
                "pydantic-core",
                "--format",
                "requirements-txt",
                "-o",
                str(staging / "requirements.txt"),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                "hf",
                "upload",
                REPO_ID,
                str(staging),
                "--repo-type",
                "space",
                "--exclude",
                "**/__pycache__/**",
                "--commit-message",
                "Sync FloorPlanBench",
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
