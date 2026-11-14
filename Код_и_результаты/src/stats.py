"""
Сводные числа отчёта (средние, доверительные интервалы, критерий Уилкоксона,
поправка Холма, порог обнаружения) считаются здесь напрямую из таблиц
результатов. Ручного переноса цифр нет — текст и данные не могут разойтись.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
RES = Path(__import__("os").environ.get("LAB6_RESULTS", ROOT / "results"))
sys.path.insert(0, str(ROOT / "src"))
from common import NNF, NNFA  # noqa: E402

RF_OOB = "Random Forest (OOB error)"
CONTROL = "Контроль: |y − медиана|"


def c(x: float, nd: int = 2) -> str:
    """Число по-русски: десятичная запятая, минус — длинный; округление
    арифметическое (0,545 → 0,55), а не банковское с ошибкой двоичного представления."""
    from decimal import ROUND_HALF_UP, Decimal
    if x != x:
        return "nan"
    q = Decimal(repr(round(float(x), 10))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return f"{q:.{nd}f}".replace(".", ",").replace("-", "−")


def holm(pvals: list[float]) -> list[float]:
    """Поправка Холма на множественные сравнения."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def k_star(df_m: pd.DataFrame) -> float:
    """Мощность, при которой средний F1 впервые достигает 0,5 (линейная
    интерполяция между узлами сетки); nan, если не достигает."""
    g = df_m.groupby("k_sigma").f1.mean().sort_index()
    k, f = g.index.values, g.values
    above = np.flatnonzero(f >= 0.5)
    if not len(above):
        return float("nan")
    j = above[0]
    if j == 0:
        return float(k[0])
    return float(k[j - 1] + (0.5 - f[j - 1]) * (k[j] - k[j - 1]) / (f[j] - f[j - 1]))


def k_star_boot(df_m: pd.DataFrame, n_boot: int = 2000, seed: int = 0):
    """Бутстреп по повторам: доверительный интервал порога обнаружимости."""
    rng = np.random.default_rng(seed)
    runs = df_m.run.unique()
    piv = df_m.pivot_table(index="run", columns="k_sigma", values="f1")
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(runs, size=len(runs), replace=True)
        m = piv.loc[pick].mean()
        tmp = pd.DataFrame({"k_sigma": m.index, "f1": m.values})
        vals.append(k_star(tmp))
    vals = np.array(vals)
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _blk_1(S: dict) -> None:
    d = pd.read_csv(RES / "exp1_magnitude.csv")
    a = pd.read_csv(RES / "exp1_anchor.csv")
    S["anchor_nnf"] = a[a.method == NNF].f1.mean()
    S["anchor_nnfa"] = a[a.method == NNFA].f1.mean()
    cls_anchor = a[~a.method.isin([NNF, NNFA, CONTROL])].groupby("method").f1.mean()
    S["anchor_best_cls_name"] = cls_anchor.idxmax()
    S["anchor_best_cls"] = cls_anchor.max()
    S["anchor_rfoob"] = a[a.method == RF_OOB].f1.mean()

    S["ctrl_max"] = d[d.method == CONTROL].groupby("k_sigma").f1.mean().max()
    mean = d.groupby(["method", "k_sigma"]).f1.mean().unstack()
    S["f1"] = mean  # таблица метод × k
    S["k_grid"] = list(mean.columns)
    S["kstar"] = {m: k_star(d[d.method == m]) for m in mean.index}
    S["kstar_ci"] = {m: k_star_boot(d[d.method == m]) for m in
                     [NNFA, NNF, RF_OOB, "Neural Network (error)"]
                     if np.isfinite(S["kstar"][m])}
    others = [m for m in mean.index if m not in (NNF, NNFA, CONTROL)]
    S["n_classical"] = len(others)
    S["cls_reach_half"] = sorted([m for m in others if mean.loc[m].max() >= 0.5],
                                 key=lambda m: S["kstar"][m])
    nonmodel = [m for m in others if m not in S["cls_reach_half"]]
    S["nonmodel_max"] = mean.loc[nonmodel].max().max()
    S["nonmodel_max_name"] = mean.loc[nonmodel].max(axis=1).idxmax()

    # NNFA против каждого из сильных конкурентов, поправка Холма по всем k
    tests = []
    for rival in [RF_OOB, "Neural Network (error)", "Random Forest (error)"]:
        for k in S["k_grid"]:
            x = d[(d.method == NNFA) & (d.k_sigma == k)].set_index("run").f1
            y = d[(d.method == rival) & (d.k_sigma == k)].set_index("run").f1
            idx = x.index.intersection(y.index)
            diff = (x[idx] - y[idx]).values
            p = 1.0 if np.allclose(diff, 0) else float(wilcoxon(x[idx], y[idx]).pvalue)
            tests.append(dict(rival=rival, k=k, diff=diff.mean(), p=p))
    t = pd.DataFrame(tests)
    t["p_holm"] = holm(t.p.tolist())
    S["tests"] = t



def _blk_2(S: dict) -> None:
    p = pd.read_csv(RES / "exp2_peeling.csv")
    S["peel"] = p.groupby(["mode", "contamination", "method"]).agg(
        f1=("f1", "mean"), precision=("precision", "mean"), recall=("recall", "mean"),
        n_detected=("n_detected", "mean"), n_out=("n_out", "first"),
        seconds=("seconds", "mean")).reset_index()
    pv = p.pivot_table(index=["mode", "contamination", "run"], columns="method", values="f1")
    peel, thr, orc = "NNFA-peel (автоостановка)", "NNFA-порог (один шаг)", "NNFA top-k (оракул)"
    S["peel_vs_thr_p"] = float(wilcoxon(pv[peel], pv[thr]).pvalue)
    S["peel_vs_orc_p"] = float(wilcoxon(pv[peel], pv[orc]).pvalue)
    S["peel_mean"], S["thr_mean"], S["orc_mean"] = pv[peel].mean(), pv[thr].mean(), pv[orc].mean()
    iso = p[(p["mode"] == "isolated") & (p.method == peel)]
    S["peel_iso_ratio"] = (iso.n_detected / iso.n_out).mean()
    S["peel_seconds"] = p[p.method == peel].seconds.mean()
    # качество самого порядка удаления: первые m удалённых при известном m
    tr = pd.read_csv(RES / "exp2_trajectories.csv")
    rows = []
    for (c_, mode_, run_), g in tr.groupby(["contamination", "mode", "run"]):
        g = g.sort_values("step")
        m_ = int(g.n_out.iloc[0])
        rows.append(dict(mode=mode_, contamination=c_, run=run_,
                         f1_budget=g.removed_is_outlier.values[:m_].mean()))
    pb = pd.DataFrame(rows)
    S["peel_budget"] = pb.groupby(["mode", "contamination"]).f1_budget.mean()
    S["peel_budget_runs"] = pb
    # время до автоостановки: траектория для рисунка длиннее, поэтому время
    # пересчитывается по среднему времени шага и номеру шага остановки
    sec = p[p.method == peel].set_index(["contamination", "mode", "run"]).seconds
    t_auto = []
    for key, g in tr.groupby(["contamination", "mode", "run"]):
        g = g.sort_values("step")
        below, stop = 0, len(g)
        for i, (z_, l_) in enumerate(zip(g.z, g.lam), start=1):
            below = 0 if z_ >= l_ else below + 1
            if below >= 3:
                stop = i
                break
        t_auto.append(sec.loc[key] / len(g) * stop)
    S["peel_seconds_auto"] = float(np.mean(t_auto))
    cl = pd.read_csv(RES / "exp2c_clean.csv")
    S["clean_mean"], S["clean_max"], S["clean_runs"] = cl.n_detected.mean(), int(cl.n_detected.max()), len(cl)



def _blk_3g(S: dict) -> None:
    g = pd.read_csv(RES / "exp3g_optimizer.csv")
    per = g.groupby(["config", "N"]).hit_top1.mean().reset_index()
    S["opt_share"] = (per.assign(ok=per.hit_top1 >= 0.5).groupby("config").ok.mean() * 100).round().astype(int).to_dict()



def _blk_3c(S: dict) -> None:
    sc = pd.read_csv(RES / "exp3c_n_scan.csv").groupby("N").f1.mean()
    S["nscan_min"], S["nscan_max"] = sc.min(), sc.max()
    S["nscan_Nmin"], S["nscan_Nmax"] = int(sc.index.min()), int(sc.index.max())



def _blk_3b(S: dict) -> None:
    xb = pd.read_csv(RES / "exp3b_xi_sensitivity.csv").groupby("xi")[["f1", "seconds", "n_nets"]].mean()
    S["xi"] = xb



def _blk_3a(S: dict) -> None:
    xa = pd.read_csv(RES / "exp3a_x_outliers.csv")
    ma = xa.groupby(["method", "n_features_hit", "k_sigma"]).f1.mean()
    S["x_nnfa_max"] = max(ma.loc[NNFA].max(), ma.loc[NNF].max())
    S["x_knn_3_4"] = ma.loc[("kNN (pyod)", 3, 4.0)]
    S["x_nnfa_3_4"] = ma.loc[(NNFA, 3, 4.0)]



def _blk_3d(S: dict) -> None:
    dq = pd.read_csv(RES / "exp3d_sample_size.csv").groupby(["method", "Q"]).f1.mean()
    S["q100"] = {m: dq.loc[(m, 100)] for m in dq.index.get_level_values(0).unique()}



def _blk_3e(S: dict) -> None:
    e = pd.read_csv(RES / "exp3e_downstream.csv")
    base = e[e.strategy == "без очистки"].set_index("split").rmse
    rows, pvals = [], []
    for s_, gg in e.groupby("strategy"):
        v = gg.set_index("split").rmse
        m_ = v.mean()
        se = v.std(ddof=1) / np.sqrt(len(v))
        pval = np.nan if s_ == "без очистки" else float(wilcoxon(v[base.index], base).pvalue)
        rows.append(dict(strategy=s_, rmse=m_, se=se, p=pval,
                         precision=gg.precision.mean(), recall=gg.recall.mean(),
                         n_removed=gg.n_removed.mean()))
    er = pd.DataFrame(rows)
    mask = er.p.notna()
    er.loc[mask, "p_holm"] = holm(er.loc[mask, "p"].tolist())
    S["down"] = er.set_index("strategy")
    S["down_n"] = e.split.nunique()


def q_min(curve: pd.Series, target: float) -> float:
    """Наименьший объём Q, при котором средний RMSE не выше target: линейная
    интерполяция по log Q между узлами сетки; nan — если не достигается."""
    q = curve.index.values.astype(float)
    r = curve.values
    ok = np.flatnonzero(r <= target)
    if not len(ok):
        return float("nan")
    j = ok[0]
    if j == 0:
        return float(q[0])
    lq0, lq1 = np.log(q[j - 1]), np.log(q[j])
    w = (r[j - 1] - target) / (r[j - 1] - r[j])
    return float(np.exp(lq0 + w * (lq1 - lq0)))


def _blk_4(S: dict) -> None:
    e = pd.read_csv(RES / "exp4_sample_size.csv")
    mean = e.groupby(["contamination", "strategy", "Q"]).rmse.mean()
    S["ss_mean"] = mean
    S["ss_runs"] = e.run.nunique()
    S["ss_sizes"] = sorted(e.Q.unique())
    q_full = max(S["ss_sizes"])
    rows = []
    for c_ in sorted(e.contamination.unique()):
        target = mean.loc[(c_, "без очистки", q_full)]       # точность без очистки на всей выборке
        for st in e.strategy.unique():
            rows.append(dict(contamination=c_, strategy=st, target=target,
                             q_min=q_min(mean.loc[(c_, st)], target)))
    qm = pd.DataFrame(rows)
    qm["ratio"] = q_full / qm.q_min
    S["ss_qmin"] = qm.set_index(["contamination", "strategy"])
    # бутстреп по повторам: доверительный интервал минимального объёма и кратности сокращения
    rng = np.random.default_rng(0)
    runs = np.sort(e.run.unique())
    piv = e.pivot_table(index=["contamination", "strategy", "run"], columns="Q", values="rmse")
    boot = []
    for _ in range(1000):
        pick = rng.choice(runs, size=len(runs), replace=True)
        for c_ in sorted(e.contamination.unique()):
            sub = piv.loc[c_]
            mb = {st: sub.loc[st].loc[pick].mean() for st in sub.index.get_level_values(0).unique()}
            target = mb["без очистки"][q_full]
            for st, curve in mb.items():
                boot.append(dict(contamination=c_, strategy=st, q_min=q_min(curve, target)))
    bt = pd.DataFrame(boot)
    bt["ratio"] = q_full / bt.q_min
    ci = bt.groupby(["contamination", "strategy"]).agg(
        q_lo=("q_min", lambda v: np.nanpercentile(v, 2.5)), q_hi=("q_min", lambda v: np.nanpercentile(v, 97.5)),
        r_lo=("ratio", lambda v: np.nanpercentile(v, 2.5)), r_hi=("ratio", lambda v: np.nanpercentile(v, 97.5)),
        reach=("q_min", lambda v: float(np.isfinite(v).mean())))
    S["ss_qmin"] = S["ss_qmin"].join(ci)
    # очистка против её отсутствия: доля пар (Q, повтор), где RMSE меньше, и парный критерий
    # Уилкоксона по всем 70 парам каждой доли загрязнения; поправка Холма по 6 сравнениям
    w = e.pivot_table(index=["contamination", "Q", "run"], columns="strategy", values="rmse")
    tests = []
    for c_ in sorted(e.contamination.unique()):
        wc = w.loc[c_]
        for st in ["NNF (m известно)", "NNFA (m известно)", "NNFA, последовательное удаление"]:
            d = wc[st] - wc["без очистки"]
            tests.append(dict(contamination=c_, strategy=st, diff=d.mean(), share_better=(d < 0).mean(),
                              n_pairs=len(d), p=float(wilcoxon(wc[st], wc["без очистки"]).pvalue)))
    tt = pd.DataFrame(tests)
    tt["p_holm"] = holm(tt.p.tolist())
    S["ss_tests"] = tt
    S["ss_better_by_q"] = (w.sub(w["без очистки"], axis=0) < 0).groupby(level=[0, 1]).mean()
    S["ss_detect"] = e[e.strategy != "без очистки"].groupby(["contamination", "strategy", "Q"])[
        ["n_removed", "m", "precision", "recall"]].mean()


BLOCKS = [_blk_1, _blk_2, _blk_3g, _blk_3c, _blk_3b, _blk_3a, _blk_3d, _blk_3e, _blk_4]


def compute() -> dict:
    """Все числа текста по таблицам results/. Блок, для которого таблицы ещё нет
    (эксперимент не посчитан), пропускается — так ноутбук можно выполнять по
    порядку с нуля: каждый раздел обращается только к уже посчитанным данным."""
    S: dict = {}
    for blk in BLOCKS:
        try:
            blk(S)
        except FileNotFoundError:
            pass
    return S

if __name__ == "__main__":
    S = compute()
    pd.set_option("display.width", 200)
    for k, v in S.items():
        print(f"== {k}\n{v}\n")
