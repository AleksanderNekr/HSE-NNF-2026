"""
Эксперимент 2 — последовательное удаление выбросов.

Сравниваются три способа отобрать выбросы по невязкам NNFA (ξ = 0,1):

  * NNFA-peel      — последовательное удаление с автоматической остановкой
                     по статистическому критерию (модуль peeling.py);
  * NNFA-порог     — одношаговый NNFA с тем же статистическим порогом;
  * NNFA top-k     — одношаговый NNFA, которому сообщено истинное число
                     выбросов (оракул; верхний ориентир, в практике недоступен).

Условия: Concrete Strength, загрязнение 1, 2 и 3%, мощность 2σ, выбросы
изолированные и кластерные (группа ближайших соседей в пространстве X,
сдвинутых в одну сторону, — случай взаимной маскировки), 15 повторов.

Результаты: results/exp2_peeling.csv (итоги), results/exp2_trajectories.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import N_JOBS, RESULTS, SMOKE, prf
from datasets import inject_y_outliers, load_concrete, load_dataset
from nnfa_core import nnfa_scores, top_k_flags
from peeling import nnfa_oneshot_threshold, nnfa_peel

CONTAMINATIONS = [0.01, 0.02, 0.03]
MODES = ["isolated", "cluster"]
K_SIGMA = 2.0
N_RUNS = 15
XI = 0.1
if SMOKE:
    CONTAMINATIONS, MODES, N_RUNS = [0.01], ["isolated", "cluster"], 1


def one_run(c: float, mode: str, run: int, ds: str = "concrete") -> tuple[list[dict], list[dict]]:
    X, y0 = load_dataset(ds)
    n_out = int(round(c * len(y0)))
    rng = np.random.default_rng(777 * run + int(c * 1000) + (0 if mode == "isolated" else 5))
    y, truth = inject_y_outliers(y0, n_out, K_SIGMA, rng, mode=mode, X=X)
    seed = 1000 * run

    t0 = time.perf_counter()
    peel = nnfa_peel(X, y, xi=XI, seed=seed, min_steps=n_out + 8)
    t_peel = time.perf_counter() - t0

    rows = []
    p, r, f = prf(peel.detected, truth)
    rows.append(dict(method="NNFA-peel (автоостановка)", n_detected=len(peel.detected),
                     precision=p, recall=r, f1=f, seconds=t_peel))

    pred = nnfa_oneshot_threshold(X, y, xi=XI, seed=seed)
    p, r, f = prf(np.flatnonzero(pred), truth)
    rows.append(dict(method="NNFA-порог (один шаг)", n_detected=int(pred.sum()),
                     precision=p, recall=r, f1=f, seconds=np.nan))

    s = nnfa_scores(X, y, xi=XI, seed=seed)
    pred = top_k_flags(s, n_out)
    p, r, f = prf(np.flatnonzero(pred), truth)
    rows.append(dict(method="NNFA top-k (оракул)", n_detected=n_out,
                     precision=p, recall=r, f1=f, seconds=np.nan))

    for row in rows:
        row.update(contamination=c, mode=mode, run=run, n_out=n_out)

    traj = [dict(contamination=c, mode=mode, run=run, n_out=n_out, step=i + 1,
                 z=z, lam=lam, removed_is_outlier=int(truth[idx]))
            for i, (z, lam, idx) in enumerate(zip(peel.z, peel.lam, peel.order))]
    return rows, traj


def main() -> None:
    t0 = time.time()
    jobs = [(c, m, r) for c in CONTAMINATIONS for m in MODES for r in range(N_RUNS)]
    out = Parallel(n_jobs=N_JOBS, verbose=2)(delayed(one_run)(*j) for j in jobs)
    pd.DataFrame([x for o, _ in out for x in o]).to_csv(RESULTS / "exp2_peeling.csv", index=False)
    pd.DataFrame([x for _, t in out for x in t]).to_csv(RESULTS / "exp2_trajectories.csv", index=False)
    print(f"exp2 готов за {(time.time() - t0) / 60:.1f} мин", flush=True)


if __name__ == "__main__":
    main()
