"""
Эксперимент 4 — снижает ли очистка методами NNF и NNFA требования к объёму
обучающей выборки.

Постановка (по направлению, сформулированному авторами методов: «необходимо
экспериментально показать, что очистка датасетов от выбросов методами NNF и
NNFA снижает требования к минимально допустимому объёму обучающих выборок»).

* Concrete Compressive Strength: в каждом повторе случайно откладывается чистая
  тестовая часть (206 наблюдений, 20 %), из остальных 824 берутся обучающие
  выборки объёма Q = 100, 150, 200, 300, 400, 600, 824 (вложенные: меньшая —
  начало большей).
* В обучающую выборку вносится доля c = 5 % или 10 % структурных выбросов
  мощностью 2σ (сдвиг к центру в пределах диапазона y).
* Стратегии подготовки данных перед обучением итоговой сети:
    - без очистки;
    - NNF: удалить m наблюдений с наибольшей ошибкой обучения одиночной сети
      (m известно — протокол препринта);
    - NNFA (ξ = 0,1): то же для ансамбля;
    - NNFA, последовательное удаление с автоостановкой (m неизвестно);
    - идеальная очистка (удалены ровно внесённые выбросы) — предел.
* Итоговая модель — персептрон (32, 16), как в §6.6; RMSE на чистой тестовой
  части усредняется по 3 инициализациям.
* 10 повторов (разные тестовые части, позиции выбросов, начальные веса).

Минимально допустимый объём выборки для заданной точности T считается по
средней кривой RMSE(Q) — это наименьший Q, при котором RMSE ≤ T (линейная
интерполяция по log Q). Расчёт — в src/stats.py.

Результат: results/exp4_sample_size.csv
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
from datasets import inject_y_outliers, load_concrete
from exp3_extensions import _rmse
from nnfa_core import nnfa_scores, top_k_flags
from peeling import nnfa_peel

SIZES = [100, 150, 200, 300, 400, 600, 824]
CONTAMINATIONS = [0.05, 0.10]
N_RUNS = 10
K_SIGMA = 2.0
N_TEST = 206
if SMOKE:
    SIZES, CONTAMINATIONS, N_RUNS = [100, 400], [0.10], 1

STRATEGIES = ["без очистки", "NNF (m известно)", "NNFA (m известно)",
              "NNFA, последовательное удаление", "идеальная очистка"]


def one(c: float, Q: int, run: int) -> list[dict]:
    X0, y0 = load_concrete()
    perm = np.random.default_rng(120_000 + run).permutation(len(y0))
    te, pool = perm[:N_TEST], perm[N_TEST:]
    tr = pool[:Q]                                   # вложенные выборки
    m = int(round(c * Q))
    rng = np.random.default_rng(130_000 + 1000 * run + Q + int(round(c * 1000)))
    Xtr, Xte, yte = X0[tr], X0[te], y0[te]
    ytr, truth = inject_y_outliers(y0[tr], m, K_SIGMA, rng)
    idx = np.arange(Q)
    rows = []

    def add(name: str, removed, seconds: float = np.nan) -> None:
        removed = np.asarray(removed, int)
        keep = np.ones(Q, bool)
        keep[removed] = False
        p, r, f = prf(removed, truth)
        t0 = time.perf_counter()
        rmse = _rmse(Xtr[keep], ytr[keep], Xte, yte, run)
        rows.append(dict(contamination=c, Q=Q, run=run, m=m, strategy=name,
                         rmse=rmse, n_removed=len(removed), precision=p, recall=r, f1=f,
                         clean_seconds=seconds, fit_seconds=time.perf_counter() - t0))

    add("без очистки", [])
    for xi, name in [(0.0, "NNF (m известно)"), (0.1, "NNFA (m известно)")]:
        t0 = time.perf_counter()
        s = nnfa_scores(Xtr, ytr, xi=xi, seed=run)
        add(name, idx[top_k_flags(s, m).astype(bool)], time.perf_counter() - t0)
    t0 = time.perf_counter()
    peel = nnfa_peel(Xtr, ytr, xi=0.1, seed=10 * run, max_frac=0.25)
    add("NNFA, последовательное удаление", peel.detected, time.perf_counter() - t0)
    add("идеальная очистка", idx[truth == 1])
    return rows


def main() -> pd.DataFrame:
    t0 = time.time()
    jobs = [(c, Q, r) for c in CONTAMINATIONS for Q in SIZES for r in range(N_RUNS)]
    # крупные выборки первыми — лучше загрузка ядер
    jobs.sort(key=lambda j: -j[1])
    out = Parallel(n_jobs=N_JOBS, verbose=2)(delayed(one)(*j) for j in jobs)
    df = pd.DataFrame([x for o in out for x in o]).sort_values(["contamination", "Q", "run"])
    df.to_csv(RESULTS / "exp4_sample_size.csv", index=False)
    print(f"exp4 готов за {(time.time() - t0) / 60:.1f} мин", flush=True)
    return df


if __name__ == "__main__":
    main()
