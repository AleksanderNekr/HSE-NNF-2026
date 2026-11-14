"""
Эксперимент 3 — собственные направления проверки методов.
Каждый сюжет закрывает ограничение, названное авторами препринта:

  3a. Выбросы во ВХОДНЫХ признаках X, а не в целевой переменной.
  3b. Чувствительность NNFA к параметру ξ: качество против вычислительной цены.
  3c. Чувствительность одиночной сети (NNF) к числу скрытых нейронов N на
      реальных данных — скан далеко за пределы интервала АКХН.
  3d. Зависимость от объёма выборки Q при фиксированной доле выбросов.
  3e. Влияние очистки обучающей выборки на точность итоговой нейросетевой
      модели на чистом отложенном множестве.

Во всех сюжетах, кроме 3a, выбросы структурные (см. datasets.inject_y_outliers):
загрязнение 2% (в 3e — 3%), мощность 2σ.

Запуск: python exp3_extensions.py [3a|3b|3c|3d|3e|all]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines import CLASSICAL_METHODS, RF_OOB, classical_scores
from common import N_JOBS, NNF, NNFA, RESULTS, SMOKE, evaluate, prf
from datasets import inject_x_outliers, inject_y_outliers, load_concrete, load_dataset
from nnfa_core import (n_range, nnfa_scores, scale_minus1_1, top_k_flags,
                       train_mlp_errors)
from peeling import nnfa_peel

R = 3 if SMOKE else None     # в режиме проверки — по 3 повтора


def _runs(n: int) -> range:
    return range(R if R else n)


# ------------------------------------------------------------------ 3a
def _run_3a(k: float, n_hit: int, run: int) -> list[dict]:
    X0, y = load_concrete()
    n_out = int(round(len(y) * 0.02))
    rng = np.random.default_rng(31_000 + 97 * run + int(k * 10) + n_hit)
    X, truth = inject_x_outliers(X0, n_out, k, rng, n_features_hit=n_hit)
    rows = []
    for xi, label in [(0.0, NNF), (0.1, NNFA)]:
        f1, auc = evaluate(nnfa_scores(X, y, xi=xi, seed=run), truth)
        rows.append(dict(k_sigma=k, n_features_hit=n_hit, run=run, method=label, f1=f1, auc=auc))
    for name, s in classical_scores(X, y, seed=run, methods=CLASSICAL_METHODS).items():
        f1, auc = evaluate(s, truth)
        rows.append(dict(k_sigma=k, n_features_hit=n_hit, run=run, method=name, f1=f1, auc=auc))
    return rows


def exp3a() -> None:
    jobs = [(k, h, r) for k in [1.0, 2.0, 3.0, 4.0] for h in [1, 3] for r in _runs(15)]
    out = Parallel(n_jobs=N_JOBS)(delayed(_run_3a)(*j) for j in jobs)
    pd.DataFrame([x for o in out for x in o]).to_csv(RESULTS / "exp3a_x_outliers.csv", index=False)


# ------------------------------------------------------------------ 3b
def _run_3b(xi: float, run: int) -> dict:
    X, y0 = load_concrete()
    n_out = int(round(len(y0) * 0.02))
    y, truth = inject_y_outliers(y0, n_out, 2.0, np.random.default_rng(42_000 + run))
    t0 = time.perf_counter()
    s = nnfa_scores(X, y, xi=xi, seed=run)
    dt = time.perf_counter() - t0
    f1, auc = evaluate(s, truth)
    return dict(xi=xi, run=run, n_nets=len(n_range(len(y), X.shape[1], 1, xi)),
                f1=f1, auc=auc, seconds=dt)


def exp3b() -> None:
    xis = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
    out = Parallel(n_jobs=N_JOBS)(delayed(_run_3b)(xi, r) for xi in xis for r in _runs(10))
    pd.DataFrame(out).to_csv(RESULTS / "exp3b_xi_sensitivity.csv", index=False)


# ------------------------------------------------------------------ 3c
def _run_3c(N: int, run: int) -> dict:
    X0, y0 = load_concrete()
    n_out = int(round(len(y0) * 0.02))
    y, truth = inject_y_outliers(y0, n_out, 2.0, np.random.default_rng(55_000 + run))
    err = train_mlp_errors(scale_minus1_1(X0), scale_minus1_1(y.reshape(-1, 1)).ravel(),
                           N, epochs=500, rng=np.random.default_rng(900 + run))
    f1, auc = evaluate(err, truth)
    return dict(N=N, run=run, f1=f1, auc=auc)


def exp3c() -> None:
    grid = [2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 144, 170, 200, 260]
    out = Parallel(n_jobs=N_JOBS)(delayed(_run_3c)(N, r) for N in grid for r in _runs(15))
    pd.DataFrame(out).to_csv(RESULTS / "exp3c_n_scan.csv", index=False)


# ------------------------------------------------------------------ 3d
def _run_3d(Q: int, run: int) -> list[dict]:
    X0, y0 = load_concrete()
    rng = np.random.default_rng(66_000 + 13 * run + Q)
    idx = rng.choice(len(y0), size=Q, replace=False)
    Xq, yq = X0[idx], y0[idx]
    y, truth = inject_y_outliers(yq, max(1, int(round(Q * 0.02))), 2.0, rng)
    rows = []
    for xi, label in [(0.0, NNF), (0.1, NNFA)]:
        f1, auc = evaluate(nnfa_scores(Xq, y, xi=xi, seed=run), truth)
        rows.append(dict(Q=Q, run=run, method=label, f1=f1, auc=auc))
    names = [RF_OOB, "Random Forest (error)", "Neural Network (error)", "LOF (with y)"]
    for name, s in classical_scores(Xq, y, seed=run, methods=names).items():
        f1, auc = evaluate(s, truth)
        rows.append(dict(Q=Q, run=run, method=name, f1=f1, auc=auc))
    return rows


def exp3d() -> None:
    jobs = [(Q, r) for Q in [100, 200, 400, 600, 800, 1030] for r in _runs(15)]
    out = Parallel(n_jobs=N_JOBS)(delayed(_run_3d)(*j) for j in jobs)
    pd.DataFrame([x for o in out for x in o]).to_csv(RESULTS / "exp3d_sample_size.csv", index=False)


# ------------------------------------------------------------------ 3e
MLP_SEEDS = 3


def _rmse(Xtr, ytr, Xte, yte, split: int) -> float:
    """RMSE, усреднённый по нескольким инициализациям итоговой сети:
    иначе разница между стратегиями очистки тонет в разбросе обучения.

    Критерий останова обучения намеренно строгий (tol=1e-7, терпение 100
    эпох): при настройках sklearn по умолчанию обучение иногда обрывается
    на плато через 200–300 эпох, и RMSE отражает сбой оптимизатора, а не
    качество данных."""
    import warnings
    warnings.filterwarnings("ignore")
    sc = StandardScaler().fit(Xtr)
    vals = []
    for s in range(MLP_SEEDS):
        m = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=3000, tol=1e-7,
                         n_iter_no_change=100, random_state=100 * split + s)
        m.fit(sc.transform(Xtr), ytr)
        vals.append(np.sqrt(np.mean((m.predict(sc.transform(Xte)) - yte) ** 2)))
    return float(np.mean(vals))


def _run_3e(split: int, ds: str = "concrete") -> list[dict]:
    X0, y0 = load_dataset(ds)
    rng = np.random.default_rng(88_000 + split)
    perm = rng.permutation(len(y0))
    n_te = int(0.2 * len(y0))
    te, tr = perm[:n_te], perm[n_te:]
    Xtr, Xte, yte = X0[tr], X0[te], y0[te]          # тестовая часть остаётся чистой
    n_out = int(round(len(tr) * 0.03))
    ytr, truth = inject_y_outliers(y0[tr], n_out, 2.0, rng)

    rows = []

    def add(name, keep_mask, removed_idx):
        p, r, f = prf(removed_idx, truth)
        rows.append(dict(split=split, strategy=name,
                         rmse=_rmse(Xtr[keep_mask], ytr[keep_mask], Xte, yte, split),
                         n_removed=len(removed_idx), precision=p, recall=r, f1=f))

    all_idx = np.arange(len(ytr))
    add("без очистки", np.ones(len(ytr), bool), [])
    add("идеальная очистка (оракул)", truth == 0, all_idx[truth == 1])

    s = nnfa_scores(Xtr, ytr, xi=0.1, seed=split)
    pred = top_k_flags(s, n_out).astype(bool)
    add("NNFA top-k (оракул k)", ~pred, all_idx[pred])

    peel = nnfa_peel(Xtr, ytr, xi=0.1, seed=10 * split)
    keep = np.ones(len(ytr), bool)
    keep[peel.detected] = False
    add("NNFA-peel (автоостановка)", keep, peel.detected)
    return rows


def exp3e() -> None:
    out = Parallel(n_jobs=N_JOBS)(delayed(_run_3e)(s) for s in _runs(30))
    pd.DataFrame([x for o in out for x in o]).to_csv(RESULTS / "exp3e_downstream.csv", index=False)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("3a", exp3a), ("3b", exp3b), ("3c", exp3c), ("3d", exp3d), ("3e", exp3e)]:
        if which in ("all", name):
            t0 = time.time()
            fn()
            print(f"{name} готов за {(time.time() - t0) / 60:.1f} мин", flush=True)
