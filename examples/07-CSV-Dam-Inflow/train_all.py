"""Train and evaluate all (model, input set) combinations of the comparison study.

Usage (from this folder, inside the ``nh`` environment)::

    python train_all.py                      # everything that is not finished yet, sequentially
    python train_all.py --models lstm gru    # subset of models
    python train_all.py --sets A             # subset of input sets
    python train_all.py --parallel 3         # up to 3 trainings at the same time on the GPU
    python train_all.py --evaluate-only      # only (re)build results/ from existing runs

Each training runs as its own ``nh-run train`` subprocess (fresh CUDA context, own output.log). The evaluation of
the best epoch (train / validation / test) and the result files below ``results/`` are produced by
:func:`common.evaluate_run`. Completed runs are never retrained unless ``--force`` is given.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import common

HERE = Path(__file__).resolve().parent
# xLSTM (parallel mLSTM kernels over 720 steps) needs ~21 GB of GPU memory -> never share the GPU with another run
EXCLUSIVE_MODELS = {"xlstm"}


def train_subprocess(model: str, input_set: str, log_dir: Path) -> float:
    cfg_file = common.config_path(model, input_set)
    env = {**os.environ, "MPLBACKEND": "Agg"}
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{common.experiment_name(model, input_set)}.log"
    t0 = time.time()
    with open(log_file, "w") as fp:
        subprocess.run([sys.executable, "-m", "neuralhydrology.nh_run", "train", "--config-file", str(cfg_file)],
                       cwd=HERE, env=env, check=True, stdout=fp, stderr=subprocess.STDOUT)
    return time.time() - t0


def run_one(model: str, input_set: str, force: bool, evaluate_only: bool, log_dir: Path) -> str:
    name = common.experiment_name(model, input_set)
    run_dir = common.find_run_dir(model, input_set)
    if run_dir is None or force:
        if evaluate_only:
            return f"{name}: no completed run, skipped"
        seconds = train_subprocess(model, input_set, log_dir)
        run_dir = common.find_run_dir(model, input_set)
        if run_dir is None:
            return f"{name}: training FAILED or incomplete (see {log_dir / name}.log)"
        print(f"{name}: trained in {seconds / 60:.1f} min -> {run_dir.name}", flush=True)
    # evaluate in a subprocess as well: eval_run resets the root logger and uses CUDA, which is not thread-safe
    # when several jobs finish at the same time in the thread pool.
    env = {**os.environ, "MPLBACKEND": "Agg"}
    code = (f"import json, common; s = common.evaluate_run({model!r}, {input_set!r}, force={force}); "
            f"print('SUMMARY_JSON ' + json.dumps(s, default=float))")
    proc = subprocess.run([sys.executable, "-c", code], cwd=HERE, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"evaluation failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
    import json
    summary = json.loads(next(l for l in proc.stdout.splitlines() if l.startswith("SUMMARY_JSON "))[len("SUMMARY_JSON "):])
    test = summary["metrics"]["test"]
    return (f"{name}: best epoch {summary['best_epoch']}/{summary['epochs_trained']}  "
            f"test NSE {test['NSE']:.3f}  KGE {test['KGE']:.3f}  ({summary['epoch_time_mean_s']:.0f} s/epoch)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=common.MODEL_ORDER, choices=common.MODEL_ORDER)
    parser.add_argument("--sets", nargs="+", default=common.SETS, choices=common.SETS)
    parser.add_argument("--parallel", type=int, default=1, help="number of concurrent trainings")
    parser.add_argument("--force", action="store_true", help="retrain even if a completed run exists")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=HERE / "runs" / "_logs")
    args = parser.parse_args()

    jobs = [(m, s) for m in args.models for s in args.sets]
    shared = [j for j in jobs if j[0] not in EXCLUSIVE_MODELS]
    exclusive = [j for j in jobs if j[0] in EXCLUSIVE_MODELS]
    t0 = time.time()
    messages = []
    if args.parallel > 1 and len(shared) > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(run_one, m, s, args.force, args.evaluate_only, args.log_dir): (m, s) for m, s in shared}
            for future in as_completed(futures):
                try:
                    msg = future.result()
                except Exception as err:  # keep going with the other runs
                    msg = f"{common.experiment_name(*futures[future])}: ERROR {type(err).__name__}: {err}"
                print(msg, flush=True)
                messages.append(msg)
    else:
        for m, s in shared:
            try:
                msg = run_one(m, s, args.force, args.evaluate_only, args.log_dir)
            except Exception as err:
                msg = f"{common.experiment_name(m, s)}: ERROR {type(err).__name__}: {err}"
            print(msg, flush=True)
            messages.append(msg)
    for m, s in exclusive:  # one at a time, nothing else on the GPU
        try:
            msg = run_one(m, s, args.force, args.evaluate_only, args.log_dir)
        except Exception as err:
            msg = f"{common.experiment_name(m, s)}: ERROR {type(err).__name__}: {err}"
        print(msg, flush=True)
        messages.append(msg)
    print(f"\nfinished {len(messages)} jobs in {(time.time() - t0) / 3600:.2f} h")
    for msg in messages:
        print(" ", msg)


if __name__ == "__main__":
    main()
