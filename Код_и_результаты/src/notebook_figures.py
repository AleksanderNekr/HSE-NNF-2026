"""
Дополнительные рисунки для ноутбука — в том же печатном стиле, что и
рисунки paper_figures.py: без встроенных заголовков, с
десятичной запятой и различимыми в ч/б сериями.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_figures as P
from paper_figures import C, MUTED, INK, W, ci95, comma, plt, _fmt_axes
from common import NNF, NNFA, pretty_name

RES, OUT = P.RES, P.ROOT / "figures"
OUT.mkdir(exist_ok=True)


def _save(fig, name: str) -> Path:
    p = OUT / name
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def auc_magnitude() -> Path:
    df = pd.read_csv(RES / "exp1_magnitude.csv")
    fig, ax = plt.subplots(figsize=(W, 2.45))
    for key, label, col, ls, mk in P.FIG1_SERIES:
        d = df[df.method == key]
        rows = [(k, *ci95(g.auc.values)) for k, g in d.groupby("k_sigma")]
        k, m, lo, hi = map(np.array, zip(*rows))
        ax.plot(k, m, color=col, ls=ls, marker=mk, label=label, lw=1.0 if mk is None else 1.4)
        ax.fill_between(k, lo, hi, color=col, alpha=0.13, lw=0)
    ax.set_xlabel("мощность выброса $k$, в долях $\\sigma_y$")
    ax.set_ylabel("ROC-AUC")
    _fmt_axes(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    return _save(fig, "nb_auc_moshchnost.png")


def anchor() -> Path:
    a = pd.read_csv(RES / "exp1_anchor.csv")
    g = a.groupby("method").f1.agg(["mean", "std"]).sort_values("mean")
    g = g[g["mean"] > 0.02]
    fig, ax = plt.subplots(figsize=(W, 0.28 * len(g) + 0.6))
    cols = [C[0] if m in (NNF, NNFA) else "#9a9a9a" for m in g.index]
    ax.barh(range(len(g)), g["mean"], xerr=g["std"], color=cols, height=0.6,
            error_kw=dict(lw=0.8, ecolor=INK))
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([pretty_name(m) for m in g.index])
    for i, (v, sd) in enumerate(zip(g["mean"], g["std"].fillna(0))):
        ax.text(min(v + sd, 0.93) + 0.02, i, comma(round(v, 3)), va="center", fontsize=8)
    ax.set_xlabel("F1 (среднее ± СКО по 30 повторам)")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    _fmt_axes(ax, y=False)
    return _save(fig, "nb_protokol_preprinta.png")


def peel_cluster() -> Path:
    tr = pd.read_csv(RES / "exp2_trajectories.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.3), sharey=True)
    for ax, mode, title in zip(axes, ["isolated", "cluster"],
                               ["(а) изолированные выбросы", "(б) кластер соседних выбросов"]):
        d = tr[tr["mode"] == mode]
        for (c, g), col, ls in zip(sorted(d.groupby("contamination")), C, ["-", "--", "-."]):
            n_out = int(g.n_out.iloc[0])
            rows = [(s, *ci95(h.z.values)) for s, h in g.groupby("step") if len(h) >= min(3, g.run.nunique())]
            s, m, lo, hi = map(np.array, zip(*rows))
            ax.plot(s / n_out, m, color=col, ls=ls, label=f"{int(c * 100)} %")
            ax.fill_between(s / n_out, lo, hi, color=col, alpha=0.13, lw=0)
        ax.axhline(float(d.lam.median()), color=INK, lw=0.8)
        ax.axvline(1, color=MUTED, lw=0.8, ls=(0, (3, 2)))
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("шаг / истинное число выбросов")
        _fmt_axes(ax)
    axes[0].set_ylabel("статистика $z$")
    axes[0].legend(title="загрязнение", fontsize=8)
    fig.tight_layout()
    return _save(fig, "nb_peel_klaster.png")


def xi_sensitivity() -> Path:
    x = pd.read_csv(RES / "exp3b_xi_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.2))
    rows = [(v, *ci95(g.f1.values)) for v, g in x.groupby("xi")]
    v, m, lo, hi = map(np.array, zip(*rows))
    axes[0].plot(v, m, color=C[0], marker="o")
    axes[0].fill_between(v, lo, hi, color=C[0], alpha=0.13, lw=0)
    axes[0].set_xlabel("ξ")
    axes[0].set_ylabel("F1")
    axes[0].set_title("(а) качество", fontsize=9, loc="left")
    t = x.groupby("xi").seconds.mean()
    axes[1].plot(t.index, t.values, color=C[1], marker="s", ls="--")
    axes[1].set_xlabel("ξ")
    axes[1].set_ylabel("время одного запуска, с")
    axes[1].set_title("(б) вычислительная цена", fontsize=9, loc="left")
    for ax in axes:
        _fmt_axes(ax)
    fig.tight_layout()
    return _save(fig, "nb_xi.png")


def n_scan() -> Path:
    s = pd.read_csv(RES / "exp3c_n_scan.csv")
    Q, Nx = 1030, 8
    n_min = Q / ((1 + math.log2(Q)) * (Nx + 1))
    n_max = (1 / (Nx + 1)) * ((Q / Nx + 1) * (Nx + 2) + 1)
    rows = [(n, *ci95(g.f1.values)) for n, g in s.groupby("N")]
    n, m, lo, hi = map(np.array, zip(*rows))
    fig, ax = plt.subplots(figsize=(W, 2.2))
    ax.axvspan(n_min, n_max, color=C[2], alpha=0.10, lw=0)
    ax.text(math.sqrt(n_min * n_max), 0.1, "интервал АКХН", ha="center", fontsize=8.5, color=INK)
    ax.plot(n, m, color=C[0], marker="o")
    ax.fill_between(n, lo, hi, color=C[0], alpha=0.13, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("число нейронов скрытого слоя N (логарифмическая шкала)")
    ax.set_ylabel("F1 одиночной сети")
    ax.set_ylim(0, 1.02)
    _fmt_axes(ax, x=False)
    return _save(fig, "nb_skan_N.png")


def sample_size() -> Path:
    d = pd.read_csv(RES / "exp3d_sample_size.csv")
    series = [(NNFA, "NNFA (ξ = 0,1)", C[0], "-", "o"),
              (NNF, "NNF (ξ = 0)", C[1], "--", "s"),
              ("Random Forest (OOB error)", "случайный лес (OOB)", C[3], "-.", "^"),
              ("Neural Network (error)", "персептрон (20, 10)", C[4], ":", "v"),
              ("LOF (with y)", "LOF по (X, y)", C[2], (0, (5, 1, 1, 1)), "D")]
    fig, ax = plt.subplots(figsize=(W, 2.3))
    for key, label, col, ls, mk in series:
        g = d[d.method == key]
        rows = [(q, *ci95(h.f1.values)) for q, h in g.groupby("Q")]
        q, m, lo, hi = map(np.array, zip(*rows))
        ax.plot(q, m, color=col, ls=ls, marker=mk, label=label)
        ax.fill_between(q, lo, hi, color=col, alpha=0.13, lw=0)
    ax.set_xlabel("объём выборки Q")
    ax.set_ylabel("F1")
    ax.set_ylim(-0.02, 1.02)
    _fmt_axes(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    return _save(fig, "nb_obyom_vyborki.png")


def downstream() -> Path:
    e = pd.read_csv(RES / "exp3e_downstream.csv")
    order = ["без очистки", "NNFA top-k (оракул k)", "NNFA-peel (автоостановка)",
             "идеальная очистка (оракул)"]
    fig, ax = plt.subplots(figsize=(W, 2.4))
    data = [e[e.strategy == o].rmse.values for o in order]
    labels = ["без очистки", "один шаг, топ-m\n(m известно)", "последовательное\nудаление",
              "идеальная очистка\n(оракул)"]
    bp = ax.boxplot(data, tick_labels=labels,
                    patch_artist=True, widths=0.5)
    for i, b in enumerate(bp["boxes"]):
        b.set(facecolor=C[i], alpha=0.35, edgecolor=C[i])
    for el in bp["medians"]:
        el.set(color=INK)
    ax.set_ylabel("RMSE на чистом тесте, МПа")
    ax.grid(axis="x", visible=False)
    _fmt_axes(ax, x=False)
    return _save(fig, "nb_ochistka_rmse.png")


if __name__ == "__main__":
    for fn in [auc_magnitude, anchor, peel_cluster, xi_sensitivity, n_scan, sample_size, downstream]:
        try:
            print(fn.__name__, "->", fn())
        except FileNotFoundError as err:
            print(fn.__name__, "пропущен:", err.filename)
