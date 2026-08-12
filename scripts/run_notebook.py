"""Execute a notebook's code cells in order and fail loudly on the first error.

A notebook that has never been run is a notebook that breaks in front of
whoever opens it. This runs one without needing Jupyter installed: cells
execute in order, in a single namespace, with the notebook's own directory
as the working directory, exactly as they would interactively.

It checks that the notebook *runs*; it does not check that the outputs are
right. Nothing is written back to the file.

    python scripts/run_notebook.py notebooks/detection_walkthrough.ipynb
    python scripts/run_notebook.py notebooks/baseline_pipeline.ipynb --cells 16

`--cells N` stops after N code cells. Needed for notebooks ending in a
live capture loop: those run until a key is pressed in a window that does
not exist under this runner, so they would block forever rather than fail.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # no display; figures render to memory and are discarded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook")
    parser.add_argument("--cells", type=int, default=None,
                        help="stop after this many code cells")
    args = parser.parse_args()

    path = os.path.abspath(args.notebook)
    os.chdir(os.path.dirname(path))
    with open(os.path.basename(path), encoding="utf-8") as handle:
        notebook = json.load(handle)

    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    limit = args.cells or len(cells)
    namespace: dict = {"__name__": "__main__"}

    for number, cell in enumerate(cells[:limit], start=1):
        source = "".join(cell["source"])
        print(f"\n----- cell {number}/{limit} -----", flush=True)
        try:
            exec(compile(source, f"<cell {number}>", "exec"), namespace)  # noqa: S102
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED cell {number}: {type(exc).__name__}: {exc}", flush=True)
            return 1

    skipped = len(cells) - limit
    suffix = f" ({skipped} not run)" if skipped else ""
    print(f"\nOK — {limit} of {len(cells)} code cells ran{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
