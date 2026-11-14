"""
Эксперимент 2c — сколько наблюдений NNFA-peel объявляет выбросами на исходных
данных без внесённых выбросов (10 запусков с разными начальными весами).
Реальные данные могут содержать собственные аномалии, поэтому это верхняя
оценка числа ложных срабатываний.

Результат: results/exp2c_clean.csv
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import N_JOBS, RESULTS, SMOKE
from datasets import load_concrete
from peeling import nnfa_peel


def one(run: int) -> dict:
    X, y = load_concrete()
    t0 = time.perf_counter()
    r = nnfa_peel(X, y, xi=0.1, seed=5000 + 100 * run)
    return dict(run=run, n_detected=len(r.detected), n_steps=r.n_steps,
                seconds=time.perf_counter() - t0)


if __name__ == "__main__":
    out = Parallel(n_jobs=N_JOBS)(delayed(one)(r) for r in range(2 if SMOKE else 10))
    df = pd.DataFrame(out)
    df.to_csv(RESULTS / "exp2c_clean.csv", index=False)
    print(df.describe().round(2))
