"""
Эксперимент 3g — откуда берётся «лотерея N».

В репозитории препринта рисунки пригодности сети (рис. 3, 6, 8) считаются
оптимизатором Rprop за 1000 эпох, а основной бенчмарк NNF/NNFA — оптимизатором
Adam за 500 эпох. Поэтому важно выяснить, как чувствительность одиночной сети
к выбору N зависит от алгоритма и длительности обучения.

Здесь на ТОМ ЖЕ наборе данных, что в ноутбуке `!!!_8_...` (гиперболический
параболоид, сетка 21×21 с удалением 90% нормальных точек, выброс в (0,0)
со значением y=0.45), скан по N повторяется в четырёх конфигурациях:
{Rprop, Adam} × {500, 1000 эпох}.

Результат: results/exp3g_optimizer.csv
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import SMOKE
from nnfa_core import scale_minus1_1, train_mlp_errors

os.environ.setdefault("OMP_NUM_THREADS", "1")
RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


def preprint_dataset():
    """Набор данных из ноутбука !!!_8 — воспроизведён построчно."""
    g = np.arange(-1.0, 1.05, 0.1)
    X1, X2 = np.meshgrid(g, g)
    Y = X1 ** 2 - X2 ** 2
    mask = (np.abs(X1) < 1e-10) & (np.abs(X2) < 1e-10)
    Y[mask] = 0.45
    pts = np.column_stack([X1.ravel(), X2.ravel(), Y.ravel()])
    lab = np.zeros(len(pts), int)
    lab[np.where(mask.ravel())[0]] = 1

    rng = np.random.RandomState(42)                 # seed авторов
    normal = np.where(lab == 0)[0]
    remove = rng.choice(normal, size=int(0.90 * len(normal)), replace=False)
    keep = np.ones(len(pts), bool)
    keep[remove] = False
    return pts[keep][:, :2], pts[keep][:, 2], int(np.where(lab[keep] == 1)[0][0])


# Полный факторный план 2×2: оптимизатор × длительность обучения. Рецензент
# справедливо указал, что при сравнении только Rprop/1000 и Adam/500 эффект
# оптимизатора смешан с эффектом числа эпох.
CONFIGS = [("Rprop, 500 эпох", "rprop", 500),
           ("Rprop, 1000 эпох", "rprop", 1000),
           ("Adam, 500 эпох", "adam", 500),
           ("Adam, 1000 эпох", "adam", 1000)]


def one(label: str, opt: str, epochs: int, N: int, run: int) -> dict:
    X, y, i = preprint_dataset()
    Xs = scale_minus1_1(X)
    ys = scale_minus1_1(y.reshape(-1, 1)).ravel()
    err = train_mlp_errors(Xs, ys, N, epochs=epochs,
                           rng=np.random.default_rng(42 + run), optimizer=opt)
    return dict(config=label, N=N, run=run,
                hit_top1=int(int(np.argmax(err)) == i))


def main() -> None:
    jobs = [(lbl, opt, ep, N, r)
            for lbl, opt, ep in CONFIGS
            for N in range(1, 81)
            for r in range(2 if SMOKE else 10)]
    print(f"запусков: {len(jobs)}", flush=True)
    t0 = time.time()
    out = Parallel(n_jobs=2, verbose=5)(delayed(one)(*j) for j in jobs)
    pd.DataFrame(out).to_csv(RESULTS / "exp3g_optimizer.csv", index=False)
    print(f"готово за {(time.time() - t0) / 60:.1f} мин", flush=True)


if __name__ == "__main__":
    main()
