"""
Эксперимент 1b — где NNFA сильнее случайного леса и почему.

Гипотеза: случайный лес не экстраполирует и у края области данных
предсказывает значения, близкие к среднему, поэтому выброс, смещённый к
центру, у края остаётся для него незаметным; нейросеть продолжает найденную
закономерность и видит такой выброс. Проверка: полнота обнаружения отдельно
для выбросов, исходное значение y которых лежит в крайних 20 % распределения,
и для остальных (мощность 1,5σ, загрязнение 2 %, 30 повторов).

Результат: results/exp1b_edge.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import RF_OOB, classical_scores
from common import N_JOBS, NNFA, RESULTS, SMOKE
from datasets import inject_y_outliers, load_concrete
from nnfa_core import nnfa_scores, top_k_flags

K, N_RUNS = 1.5, (3 if SMOKE else 30)


def one(run: int) -> list[dict]:
    X, y0 = load_concrete()
    n_out = int(round(0.02 * len(y0)))
    rng = np.random.default_rng(10_000 * run + int(round(K * 100)))   # те же выбросы, что в exp1
    y, truth = inject_y_outliers(y0, n_out, K, rng, mode="isolated")
    q = np.searchsorted(np.sort(y0), y0) / len(y0)                     # квантиль исходного y
    edge = (q < 0.1) | (q >= 0.9)
    scores = {NNFA: nnfa_scores(X, y, xi=0.1, seed=run),
              RF_OOB: classical_scores(X, y, seed=run, methods=[RF_OOB])[RF_OOB]}
    rows = []
    for m, s in scores.items():
        hit = top_k_flags(s, n_out).astype(bool)
        for zone, mask in [("край (крайние 20 % по y)", edge), ("внутренняя область", ~edge)]:
            sel = truth.astype(bool) & mask
            if sel.sum():
                rows.append(dict(run=run, method=m, zone=zone, n=int(sel.sum()),
                                 recall=float(hit[sel].mean())))
    return rows


if __name__ == "__main__":
    out = Parallel(n_jobs=N_JOBS)(delayed(one)(r) for r in range(N_RUNS))
    df = pd.DataFrame([x for o in out for x in o])
    df.to_csv(RESULTS / "exp1b_edge.csv", index=False)
    w = df.assign(w=df.recall * df.n).groupby(["zone", "method"])[["w", "n"]].sum()
    print((w.w / w.n).round(3))
