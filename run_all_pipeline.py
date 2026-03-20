from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_step(args: list[str], cwd: Path) -> None:
    print("Running:", " ".join(args))
    result = subprocess.run(args, cwd=str(cwd), check=False, text=True)
    if result.returncode != 0:
        raise SystemExit(f"Step failed with code {result.returncode}: {' '.join(args)}")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    # Use the currently active Python interpreter so the script is portable
    # across different machines and virtual environments.
    py = sys.executable

    if not py:
        raise SystemExit("No active Python interpreter found (sys.executable is empty).")

    run_step([py, "run_pipeline.py"], project_root)
    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
