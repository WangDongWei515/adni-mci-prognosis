"""
返修后全量重跑（人口学关联修复之后）：GBM 各阶段顺序跑；神经网络先调参，其余阶段并行（最多 5 个进程，每个 2 线程）。
lightgbm 与 torch 不能同进程，所以全部以子进程方式调用 run_gbm_v2.py / run_neural_v2.py。
用法：.venv/bin/python scripts/rerun_all_v2.py      日志：results/logs/rerun/
"""
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results/logs/rerun"
LOG.mkdir(parents=True, exist_ok=True)
PY = sys.executable
NN_JOBS = ["primary", "cv 0", "cv 1", "cv 2", "ablation A", "ablation B", "ablation C", "curated 0", "curated 1",
           "curated 2", "temporal", "sensitivity all_cause", "sensitivity wide", "sensitivity absorbing"]
GBM_JOBS = ["tune", "primary", "cv", "ablation", "curated", "temporal", "sensitivity"]


def run(script, args):
    name = f"{Path(script).stem.split('_')[1]}_{'_'.join(args)}"
    t0 = time.time()
    with open(LOG / f"{name}.log", "w") as fh:
        rc = subprocess.run([PY, script, *args], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    msg = f"{'done' if rc == 0 else 'FAIL'} {name} {time.time() - t0:.0f}s"
    print(msg, flush=True)
    return rc


def gbm_chain():
    for s in GBM_JOBS:
        if run("scripts/run_gbm_v2.py", [s]):
            return 1
    return 0


def main():
    with ThreadPoolExecutor(max_workers=6) as ex:
        g = ex.submit(gbm_chain)
        if run("scripts/run_neural_v2.py", ["tune"]):
            return
        futs = [ex.submit(run, "scripts/run_neural_v2.py", j.split()) for j in NN_JOBS]
        rcs = [f.result() for f in futs] + [g.result()]
    print("ALL DONE" if not any(rcs) else "SOME FAILED", flush=True)


if __name__ == "__main__":
    main()
