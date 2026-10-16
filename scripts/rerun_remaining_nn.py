"""
续跑神经网络剩余阶段（上一次全量重跑在会话结束时被中断）。已有输出的阶段由 run_neural_v2.py 自动跳过。
用法：nohup .venv/bin/python scripts/rerun_remaining_nn.py > results/logs/rerun/driver_nn2.log 2>&1 &
"""
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results/logs/rerun"
JOBS = ["curated 0", "curated 1", "curated 2", "ablation A", "ablation B", "ablation C", "temporal",
        "sensitivity all_cause", "sensitivity wide", "sensitivity absorbing"]


def run(args):
    name = "neural_" + "_".join(args)
    t0 = time.time()
    with open(LOG / f"{name}.log", "w") as fh:
        rc = subprocess.run([sys.executable, "scripts/run_neural_v2.py", *args], cwd=ROOT, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    print(f"{'done' if rc == 0 else 'FAIL'} {name} {time.time() - t0:.0f}s", flush=True)
    return rc


with ThreadPoolExecutor(max_workers=6) as ex:
    rcs = list(ex.map(run, [j.split() for j in JOBS]))
print("ALL DONE" if not any(rcs) else "SOME FAILED", flush=True)
