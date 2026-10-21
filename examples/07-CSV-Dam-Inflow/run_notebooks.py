"""Execute the study notebooks in place (and export html) with the ``nh`` kernel.

Usage::

    python run_notebooks.py                 # all notebooks 01 ... 13 in order
    python run_notebooks.py 01_LSTM.ipynb   # a subset
    python run_notebooks.py --html          # also write <name>.html next to each notebook

The notebooks skip training when a completed run exists (see ``train_all.py``), so executing them after
``train_all.py`` only evaluates, plots and writes ``results/`` and ``figures/``.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="*")
    parser.add_argument("--html", action="store_true")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    notebooks = [HERE / n for n in args.notebooks] or sorted(HERE.glob("[0-9][0-9]_*.ipynb"))
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}  # Agg would suppress the inline figures
    for nb in notebooks:
        t0 = time.time()
        cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
               f"--ExecutePreprocessor.timeout={args.timeout}", "--ExecutePreprocessor.kernel_name=nh", str(nb)]
        print(f"executing {nb.name} ...", flush=True)
        proc = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr[-6000:])
            print(f"FAILED {nb.name} after {(time.time() - t0) / 60:.1f} min", flush=True)
            continue
        print(f"done {nb.name} in {(time.time() - t0) / 60:.1f} min", flush=True)
        if args.html:
            subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "html", str(nb)], cwd=HERE, env=env,
                           capture_output=True, text=True)


if __name__ == "__main__":
    main()
