"""
Верификация исследования: код, протокол и числа.

  1. Воспроизводимость при фиксированных зерно-значениях.
  2. Корректность метрик (сверка с sklearn).
  3. Эквивалентность NumPy-реализации оригинальной реализации на PyTorch
     (Adam и Rprop, при одинаковых начальных весах).
  4. Корректность протокола структурных выбросов: значения остаются внутри
     диапазона y, контрольный скорер работает на уровне случайного угадывания.
  5. Корректность критерия остановки: порог λ(n) совпадает с табличным
     квантилем, на чистых данных алгоритм почти ничего не удаляет.
  6. Одинаковость данных для всех методов внутри повтора и полнота повторов.
  7. Воспроизведение опубликованного числа препринта (протокол [7]).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import NNF, RESULTS
from datasets import _structural_shift, inject_y_outliers, load_concrete
from nnfa_core import f1_at_k, nnfa_scores, scale_minus1_1, top_k_flags
from peeling import critical_value, nnfa_peel

ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    ok &= bool(cond)
    print(f"[{'OK ' if cond else 'СБОЙ'}] {name}" + (f" — {detail}" if detail else ""))


X, y0 = load_concrete()
y, truth = inject_y_outliers(y0, 21, 2.0, np.random.default_rng(0))

# 1 ----------------------------------------------------------------------
s1 = nnfa_scores(X, y, xi=0.05, seed=7)
s2 = nnfa_scores(X, y, xi=0.05, seed=7)
check("воспроизводимость при одном seed", np.array_equal(s1, s2))
check("разные seed дают разные результаты", not np.allclose(s1, nnfa_scores(X, y, xi=0.05, seed=8)))

# 2 ----------------------------------------------------------------------
check("F1@k совпадает со sklearn",
      abs(f1_at_k(s1, truth) - f1_score(truth, top_k_flags(s1, 21))) < 1e-12)
rnd = np.random.default_rng(1).normal(size=len(y))
check("случайный скор даёт ROC-AUC около 0,5", abs(roc_auc_score(truth, rnd) - 0.5) < 0.15,
      f"{roc_auc_score(truth, rnd):.3f}")

# 3 ----------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import nnfa_core as core

    def compare(opt_name: str, epochs: int = 200, N: int = 12) -> tuple[float, bool]:
        Xs, ys = scale_minus1_1(X), scale_minus1_1(y.reshape(-1, 1)).ravel()
        torch.manual_seed(5)
        m = nn.Sequential(nn.Linear(8, N), nn.Tanh(), nn.Linear(N, 1)).double()
        W = [m[0].weight.detach().numpy().T.copy(), m[0].bias.detach().numpy().copy(),
             m[2].weight.detach().numpy().T.copy(), m[2].bias.detach().numpy().copy()]
        opt = (torch.optim.Adam(m.parameters(), lr=0.01) if opt_name == "adam"
               else torch.optim.Rprop(m.parameters(), lr=0.01))
        Xt = torch.tensor(Xs)
        yt = torch.tensor(ys.reshape(-1, 1))
        for _ in range(epochs):
            opt.zero_grad()
            nn.MSELoss()(m(Xt), yt).backward()
            opt.step()
        with torch.no_grad():
            e_t = ((m(Xt).numpy().ravel() - ys) ** 2)
        calls = iter([(W[0], W[1]), (W[2], W[3])])
        orig = core._torch_linear_init
        core._torch_linear_init = lambda *_: next(calls)
        try:
            e_n = core.train_mlp_errors(Xs, ys, N, epochs=epochs, optimizer=opt_name)
        finally:
            core._torch_linear_init = orig
        return float(np.abs(e_t - e_n).max() / e_t.max()), int(np.argmax(e_t)) == int(np.argmax(e_n))

    for o in ["adam", "rprop"]:
        rel, same = compare(o)
        check(f"NumPy ≡ PyTorch, {o}", rel < 1e-10 and same, f"макс. относит. отклонение {rel:.1e}")
except ImportError:
    print("[ПРОПУСК] PyTorch не установлен — сверка с оригиналом не выполнена")

# 4 ----------------------------------------------------------------------
inside = True
for k in [0.5, 1.0, 2.0, 3.0]:
    for r in range(5):
        yy, tt = inject_y_outliers(y0, 21, k, np.random.default_rng(r))
        inside &= (yy.min() >= y0.min() - 1e-9) and (yy.max() <= y0.max() + 1e-9)
        inside &= np.allclose(np.abs(yy - y0)[tt == 1], k * y0.std())
check("выбросы остаются в диапазоне y и имеют заданную мощность", inside)
d1 = pd.read_csv(RESULTS / "exp1_magnitude.csv")
ctrl = d1[d1.method.str.startswith("Контроль")].groupby("k_sigma").f1.mean()
check("контрольный скорер |y − медиана| не находит выбросы при k ≤ 2", ctrl.loc[:2.0].max() < 0.1,
      f"max F1 = {ctrl.loc[:2.0].max():.3f}")

# 5 ----------------------------------------------------------------------
check("λ(n) — квантиль нормального закона с поправкой Бонферрони",
      abs(critical_value(1000) - norm.ppf(1 - 0.05 / 2000)) < 1e-12, f"λ(1000) = {critical_value(1000):.3f}")
clean = nnfa_peel(X, y0, xi=0.0, seed=3)
check("на исходных (незагрязнённых) данных последовательное удаление почти ничего не удаляет",
      len(clean.detected) <= 10, f"удалено {len(clean.detected)} из 1030")

yy, _ = inject_y_outliers(y0, 10, 2.0, np.random.default_rng(11))
r0 = nnfa_peel(X, yy, xi=0.0, seed=5, min_steps=0)
r1 = nnfa_peel(X, yy, xi=0.0, seed=5, min_steps=25)
check("решение последовательного удаления не зависит от длины траектории для рисунков (min_steps)",
      r0.detected == r1.detected, f"объявлено {len(r0.detected)} выбросов в обоих случаях")
from peeling import decide_online
tr = pd.read_csv(RESULTS / "exp2_trajectories.csv")
pe = pd.read_csv(RESULTS / "exp2_peeling.csv")
pe = pe[pe.method.str.startswith("NNFA-peel")].set_index(["contamination", "mode", "run"])
same = all(decide_online(g.sort_values("step").z.tolist(), g.sort_values("step").lam.tolist())
           == pe.loc[key, "n_detected"] for key, g in tr.groupby(["contamination", "mode", "run"]))
check("итоги эксп. 2 совпадают с онлайн-решением по сохранённым траекториям", same)

# 6 ----------------------------------------------------------------------
n_m = d1.method.nunique()
check("в эксп. 1 каждая точка кривой — 30 повторов", d1.groupby(["method", "k_sigma"]).size().eq(30).all())
check("внутри повтора все методы оценены на одних данных",
      d1.groupby(["k_sigma", "run"]).size().eq(n_m).all(), f"{n_m} методов")

e4 = RESULTS / "exp4_sample_size.csv"
if e4.exists():
    d4 = pd.read_csv(e4)
    full = d4.groupby(["contamination", "Q", "strategy"]).size().eq(d4.run.nunique()).all()
    same_m = d4.groupby(["contamination", "Q", "run"]).m.nunique().eq(1).all()
    check("в эксп. 4 каждая точка — все повторы, все стратегии на одних и тех же данных", full and same_m,
          f"{d4.run.nunique()} повторов × {d4.strategy.nunique()} стратегий")

# 7 ----------------------------------------------------------------------
a = pd.read_csv(RESULTS / "exp1_anchor.csv")
f_nnf = a[a.method == NNF].f1.mean()
check("протокол препринта: F1 метода NNF близок к опубликованному 0,760",
      abs(f_nnf - 0.760) < 0.05, f"получено {f_nnf:.3f}")

print("\nИТОГ:", "все проверки пройдены" if ok else "ЕСТЬ СБОИ")
if not ok:
    raise SystemExit(1)
