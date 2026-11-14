"""
Числа итогового отчёта по трём наборам данных. Всё считается из таблиц results/
(и из results/ ноутбука «Очистка_и_объём_выборки» для эксперимента с объёмом выборки).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
SS_RES = Path(os.environ.get("LAB6_SS_RESULTS", ROOT.parent / "Очистка_и_объём_выборки" / "results"))
sys.path.insert(0, str(ROOT / "src"))
from common import NNF, NNFA  # noqa: E402
from stats import CONTROL, RF_OOB, c, holm, k_star, k_star_boot  # noqa: E402

MLP = "Neural Network (error)"
# классические методы, использующие модель зависимости y от X
MODEL_BASED = {RF_OOB, "Random Forest (error)", MLP, "Combined (RF+AE)"}
DS = ["concrete", "california", "friedman"]
DS_NAME = {"concrete": "Concrete", "california": "California Housing", "friedman": "функция Фридмана"}
PEEL, THR, ORC = "NNFA-peel (автоостановка)", "NNFA-порог (один шаг)", "NNFA top-k (оракул)"


def _path(ds: str, kind: str) -> Path:
    if ds == "concrete":
        return RES / {"magnitude": "exp1_magnitude.csv", "peeling": "exp2_peeling.csv",
                      "trajectories": "exp2_trajectories.csv", "downstream": "exp3e_downstream.csv"}[kind]
    return RES / f"exp5_{ds}_{kind}.csv"


def magnitude(ds: str) -> dict:
    d = pd.read_csv(_path(ds, "magnitude"))
    mean = d.groupby(["method", "k_sigma"]).f1.mean().unstack()
    auc = d.groupby(["method", "k_sigma"]).auc.mean().unstack()
    ks = {m: k_star(d[d.method == m]) for m in mean.index}
    ci = {m: k_star_boot(d[d.method == m]) for m in [NNFA, NNF, RF_OOB, MLP] if np.isfinite(ks[m])}
    others = [m for m in mean.index if m not in (NNF, NNFA, CONTROL)]
    reach = sorted([m for m in others if mean.loc[m].max() >= 0.5], key=lambda m: ks[m])
    nonmodel = [m for m in others if m not in MODEL_BASED]
    tests = []
    for rival in [RF_OOB, MLP]:
        for k in mean.columns:
            x = d[(d.method == NNFA) & (d.k_sigma == k)].set_index("run").f1
            y = d[(d.method == rival) & (d.k_sigma == k)].set_index("run").f1
            diff = (x - y).values
            p = 1.0 if np.allclose(diff, 0) else float(wilcoxon(x, y).pvalue)
            tests.append(dict(rival=rival, k=k, diff=diff.mean(), p=p))
    t = pd.DataFrame(tests)
    t["p_holm"] = holm(t.p.tolist())
    return dict(raw=d, f1=mean, auc=auc, kstar=ks, kstar_ci=ci, reach=reach,
                nonmodel_max=mean.loc[nonmodel].max().max(),
                nonmodel_max_name=mean.loc[nonmodel].max(axis=1).idxmax(),
                ctrl_le2=mean.loc[CONTROL].loc[:2.0].max(), tests=t)


def peeling(ds: str) -> dict:
    p = pd.read_csv(_path(ds, "peeling"))
    agg = p.groupby(["mode", "contamination", "method"]).agg(
        f1=("f1", "mean"), precision=("precision", "mean"), recall=("recall", "mean"),
        n_detected=("n_detected", "mean"), n_out=("n_out", "first"), seconds=("seconds", "mean"))
    tr = pd.read_csv(_path(ds, "trajectories"))
    bud = {}
    for (c_, mode_), g in tr.groupby(["contamination", "mode"]):
        vals = []
        for _, h in g.groupby("run"):
            h = h.sort_values("step")
            m_ = int(h.n_out.iloc[0])
            vals.append(h.removed_is_outlier.iloc[:m_].sum() / m_)
        bud[(mode_, c_)] = float(np.mean(vals))
    pv = p.pivot_table(index=["mode", "contamination", "run"], columns="method", values="f1")
    iso = pv.loc["isolated"]
    return dict(agg=agg, budget=bud,
                p_iso_thr=float(wilcoxon(iso[PEEL], iso[THR]).pvalue),
                p_iso_orc=float(wilcoxon(iso[PEEL], iso[ORC]).pvalue) if not np.allclose(iso[PEEL], iso[ORC]) else 1.0,
                zero_cluster=(p[(p["mode"] == "cluster") & (p.method == PEEL)].groupby("contamination")
                              .n_detected.apply(lambda v: int((v == 0).sum()))).to_dict())


def downstream(ds: str) -> pd.DataFrame:
    e = pd.read_csv(_path(ds, "downstream"))
    w = e.pivot(index="split", columns="strategy", values="rmse")
    out = e.groupby("strategy")[["rmse", "n_removed", "precision", "recall"]].mean()
    out["se"] = e.groupby("strategy").rmse.std() / np.sqrt(e.split.nunique())
    ps = {s: float(wilcoxon(w[s], w["без очистки"]).pvalue) for s in w.columns if s != "без очистки"}
    adj = dict(zip(ps, holm(list(ps.values()))))
    out["p_holm"] = pd.Series(adj)
    out["share_better"] = pd.Series({s: float((w[s] < w["без очистки"]).mean()) for s in w.columns})
    base, ideal = out.loc["без очистки", "rmse"], out.loc["идеальная очистка (оракул)", "rmse"]
    out["effect_share"] = (base - out.rmse) / (base - ideal)
    return out


def sample_size() -> dict:
    """Результаты ноутбука «Очистка_и_объём_выборки»."""
    ratio = pd.read_csv(SS_RES / "kratnost_qmax.csv")
    tests = pd.read_csv(SS_RES / "znachimost.csv")
    ex = pd.read_csv(SS_RES / "experiment.csv")
    return dict(ratio=ratio, tests=tests, ex=ex)


def compute() -> dict:
    R = {"mag": {}, "peel": {}, "down": {}}
    for ds in DS:
        for key, fn in [("mag", magnitude), ("peel", peeling), ("down", downstream)]:
            try:
                R[key][ds] = fn(ds)
            except FileNotFoundError:
                pass
    try:
        R["ss"] = sample_size()
    except FileNotFoundError:
        pass
    return R


if __name__ == "__main__":
    R = compute()
    for ds, m in R["mag"].items():
        print(ds, {k: round(v, 3) for k, v in m["kstar"].items() if v == v}, m["reach"],
              round(m["nonmodel_max"], 3), m["nonmodel_max_name"])
    for ds, p in R["peel"].items():
        print(ds); print(p["agg"].round(3)); print(p["budget"]); print(p["zero_cluster"])
    for ds, d in R["down"].items():
        print(ds); print(d.round(3))
