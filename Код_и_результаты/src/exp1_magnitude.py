"""
Эксперимент 1 — зависимость качества от мощности выбросов.

Постановка. Concrete Compressive Strength, 2% наблюдений (21 из 1030)
получают структурный выброс мощностью k = |Δy| / σ_y: значение y сдвигается
к центру распределения и остаётся внутри наблюдаемого диапазона, поэтому
выброс не выделяется ни по одному признаку отдельно. Сетка k: 0,25σ … 3σ,
30 независимых повторов на каждую точку (свои позиции выбросов и своя
инициализация сетей). Все методы в пределах повтора получают одни и те же
данные.

Точка привязки. Отдельно воспроизводится протокол препринта (20 крайних
значений y заменяются на (max+min)/2), чтобы сверить абсолютный уровень
качества с опубликованными числами.

Результаты: results/exp1_magnitude.csv, results/exp1_anchor.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines import CLASSICAL_METHODS, CONTROL, classical_scores
from common import N_JOBS, NNF, NNFA, RESULTS, SMOKE, evaluate
from datasets import inject_preprint_scenario, inject_y_outliers, load_concrete, load_dataset
from nnfa_core import nnfa_scores

K_VALUES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
N_RUNS = 30
METHODS_CLS = CLASSICAL_METHODS + [CONTROL]
if SMOKE:
    K_VALUES, N_RUNS = [0.5, 2.0], 2


def score_all(X, y, truth, run) -> list[tuple[str, float, float]]:
    out = []
    for xi, label in [(0.0, NNF), (0.1, NNFA)]:
        out.append((label, *evaluate(nnfa_scores(X, y, xi=xi, seed=run), truth)))
    for name, s in classical_scores(X, y, seed=run, methods=METHODS_CLS).items():
        out.append((name, *evaluate(s, truth)))
    return out


def run_magnitude(k: float, run: int, ds: str = "concrete") -> list[dict]:
    X, y0 = load_dataset(ds)
    n_out = int(round(0.02 * len(y0)))
    rng = np.random.default_rng(10_000 * run + int(round(k * 100)))
    y, truth = inject_y_outliers(y0, n_out, k, rng, mode="isolated")
    return [dict(k_sigma=k, run=run, method=m, f1=f1, auc=auc)
            for m, f1, auc in score_all(X, y, truth, run)]


def run_anchor(run: int) -> list[dict]:
    X, y0 = load_concrete()
    y, truth = inject_preprint_scenario(y0, 20)
    # в протоколе препринта позиции выбросов фиксированы, повторы отличаются
    # только инициализацией сетей и случайностью классических методов
    return [dict(run=run, method=m, f1=f1, auc=auc)
            for m, f1, auc in score_all(X, y, truth, run)]


def main() -> None:
    t0 = time.time()
    jobs = [(k, r) for k in K_VALUES for r in range(N_RUNS)]
    out = Parallel(n_jobs=N_JOBS, verbose=2)(delayed(run_magnitude)(*j) for j in jobs)
    pd.DataFrame([x for o in out for x in o]).to_csv(RESULTS / "exp1_magnitude.csv", index=False)

    out = Parallel(n_jobs=N_JOBS)(delayed(run_anchor)(r) for r in range(N_RUNS))
    pd.DataFrame([x for o in out for x in o]).to_csv(RESULTS / "exp1_anchor.csv", index=False)
    print(f"exp1 готов за {(time.time() - t0) / 60:.1f} мин", flush=True)


if __name__ == "__main__":
    main()
