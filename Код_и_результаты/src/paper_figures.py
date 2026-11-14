"""
Рисунки отчёта: печатное качество, строго по правилам оформления.

* Никаких заголовков внутри рисунка — подпись «Рисунок N.» ставится в тексте.
* Шрифт с засечками, метрически совместимый с Times New Roman (как в тексте).
* Десятичная запятая на осях и в подписях — как принято в русском тексте.
* Серии различимы и в чёрно-белой печати: у каждой свой маркер и тип линии.
* Ширина под поле страницы шаблона (16 см), разрешение 300 dpi.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

from common import NNF, NNFA

ROOT = Path(__file__).resolve().parent.parent
RES = Path(__import__("os").environ.get("LAB6_RESULTS", ROOT / "results"))
OUT = ROOT / "figures_paper"
OUT.mkdir(exist_ok=True)

W = 6.3                              # дюймов: ширина поля страницы шаблона
INK, GRID, MUTED = "#111111", "#dedede", "#7a7a7a"
# проверенная на цветовую слепоту категориальная палитра, фиксированный порядок
C = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948", "#008300"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",          # формулы в гарнитуре типа Times
    "font.size": 10.5,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def comma(x, _pos=None) -> str:
    """Число с десятичной запятой без лишних нулей: 0.5 → «0,5», 1.0 → «1»."""
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", ",").replace("-", "−")


def ci95(v: np.ndarray) -> tuple[float, float, float]:
    v = np.asarray(v, float)
    m = float(v.mean())
    if len(v) < 2:
        return m, m, m
    h = float(student_t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return m, m - h, m + h


def _fmt_axes(ax, x=True, y=True):
    if x:
        ax.xaxis.set_major_formatter(FuncFormatter(comma))
    if y:
        ax.yaxis.set_major_formatter(FuncFormatter(comma))


# ------------------------------------------------------------------ рисунок 1
FIG1_SERIES = [
    (NNFA, "NNFA (ξ = 0,1)", C[0], "-", "o"),
    (NNF, "NNF (ξ = 0)", C[1], "--", "s"),
    ("Random Forest (OOB error)", "случайный лес (OOB)", C[3], "-.", "^"),
    ("Neural Network (error)", "персептрон (20, 10)", C[4], ":", "v"),
    ("LOF (with y)", "LOF по (X, y)", C[2], (0, (6, 2)), "D"),
    ("Контроль: |y − медиана|", "контроль |y − медиана|", MUTED, (0, (1, 1)), "x"),
]


def fig_magnitude() -> Path:
    df = pd.read_csv(RES / "exp1_magnitude.csv")
    fig, ax = plt.subplots(figsize=(W * 0.72, 2.45))
    ax.axhline(0.5, color="#999999", lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.text(3.02, 0.47, "F1 = 0,5", fontsize=9, color="#555555", ha="right", va="top")
    for key, label, col, ls, mk in FIG1_SERIES:
        d = df[df.method == key]
        if d.empty:
            continue
        rows = [(k, *ci95(g.f1.values)) for k, g in d.groupby("k_sigma")]
        k, m, lo, hi = map(np.array, zip(*rows))
        ax.plot(k, m, color=col, ls=ls, marker=mk, label=label,
                lw=1.0 if key.startswith("Контроль") else 1.4, zorder=3 if key == NNFA else 2)
        ax.fill_between(k, lo, hi, color=col, alpha=0.13, lw=0)
    ax.set_xlabel("мощность выброса $k$, в долях $\\sigma_y$")
    ax.set_ylabel("F1")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0.2, 3.05)
    ax.set_xticks([0.25, 0.5, 1, 1.5, 2, 2.5, 3])
    _fmt_axes(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), handlelength=2.6, fontsize=9.5)
    p = OUT / "ris1_moshchnost.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ рисунок 2
def fig_peeling() -> Path:
    tr = pd.read_csv(RES / "exp2_trajectories.csv")
    tr = tr[tr["mode"] == "isolated"]
    fig, ax = plt.subplots(figsize=(W, 2.0))
    styles = [(C[0], "-", "o"), (C[1], "--", "s"), (C[3], "-.", "^")]
    for (c, g), (col, ls, mk) in zip(sorted(tr.groupby("contamination")), styles):
        n_out = int(g.n_out.iloc[0])
        rows = [(s, *ci95(h.z.values)) for s, h in g.groupby("step") if len(h) >= min(3, g.run.nunique())]
        s, m, lo, hi = map(np.array, zip(*rows))
        rel = s / n_out
        ax.plot(rel, m, color=col, ls=ls, marker=mk, markevery=max(1, len(rel) // 12),
                label=f"{comma(c * 100)} % выбросов ($m$ = {n_out})")
        ax.fill_between(rel, lo, hi, color=col, alpha=0.13, lw=0)
    lam = float(tr.lam.median())
    ax.axhline(lam, color=INK, lw=0.9)
    xmax = ax.get_xlim()[1]
    ax.text(xmax * 0.985, lam - 0.07, f"$\\lambda(n)$ ≈ {comma(round(lam, 2))} ($\\alpha$ = 0,05)",
            fontsize=9.5, color=INK, va="top", ha="right")
    ax.axvline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.set_xlabel("номер шага удаления, отнесённый к истинному числу выбросов")
    ax.set_ylabel("статистика $z$")
    _fmt_axes(ax)
    ax.legend(loc="upper right", fontsize=9.5)
    p = OUT / "ris2_ostanovka.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ рисунок 3
def fig_optimizer() -> Path:
    df = pd.read_csv(RES / "exp3g_optimizer.csv")
    order = ["Rprop, 500 эпох", "Rprop, 1000 эпох", "Adam, 500 эпох", "Adam, 1000 эпох"]
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.75), sharey=True)
    for ax, cfg in zip(axes, order):
        a = df[df.config == cfg].groupby("N").hit_top1.mean()
        good = a.values >= 0.5
        ax.bar(a.index[good], a.values[good], width=0.9, color="#3a3a3a", lw=0)
        ax.bar(a.index[~good], a.values[~good], width=0.9, color="#bdbdbd", lw=0)
        ax.axhline(0.5, color=C[4], lw=0.8, ls=(0, (3, 2)))
        share = int(round(100 * good.mean()))
        alg, ep = cfg.replace("Rprop", "RPROP").split(", ")
        ax.set_title(f"{alg}, {ep}\nпригодна: {share} %", fontsize=9.5, pad=3)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xticks([0, 40, 80])
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("$N$", labelpad=1)
        _fmt_axes(ax, x=False)
    axes[0].set_ylabel("доля успехов")
    fig.tight_layout(w_pad=0.6)
    p = OUT / "ris3_optimizator.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ рисунок 4
FIG4_SERIES = [
    (NNFA, "NNFA (ξ = 0,1)", C[0], "-", "o"),
    (NNF, "NNF (ξ = 0)", C[1], "--", "s"),
    ("kNN (pyod)", "kNN", C[3], "-.", "^"),
    ("LOF (pyod)", "LOF", C[2], ":", "D"),
    ("Random Forest (OOB error)", "случайный лес (OOB)", C[4], (0, (5, 1, 1, 1)), "v"),
]


def fig_x_outliers() -> Path:
    df = pd.read_csv(RES / "exp3a_x_outliers.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.3), sharey=True)
    for ax, h in zip(axes, [1, 3]):
        d = df[df.n_features_hit == h]
        for key, label, col, ls, mk in FIG4_SERIES:
            g = d[d.method == key]
            rows = [(k, *ci95(v.f1.values)) for k, v in g.groupby("k_sigma")]
            k, m, lo, hi = map(np.array, zip(*rows))
            ax.plot(k, m, color=col, ls=ls, marker=mk, label=label)
            ax.fill_between(k, lo, hi, color=col, alpha=0.13, lw=0)
        ax.set_title(f"({'а' if h == 1 else 'б'}) искажён {h} признак{'' if h == 1 else 'а'}",
                     fontsize=9, loc="left", pad=3)
        ax.set_xlabel("сдвиг признака, в долях σ")
        ax.set_ylim(-0.02, 1.02)
        _fmt_axes(ax)
    axes[0].set_ylabel("F1")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(w_pad=0.8)
    p = OUT / "ris4_vybrosy_x.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ рисунок 4
FIG5_SERIES = [
    ("без очистки", "без очистки", INK, "-", "o"),
    ("NNF (m известно)", "очистка NNF ($m$ известно)", C[1], "--", "s"),
    ("NNFA (m известно)", "очистка NNFA ($m$ известно)", C[0], "-", "D"),
    ("NNFA, последовательное удаление", "NNFA, послед. удаление ($m$ неизвестно)", C[3], "-.", "^"),
    ("идеальная очистка", "идеальная очистка", C[2], (0, (1, 1.5)), "v"),
]


def fig_sample_size_rmse() -> Path:
    """RMSE итоговой сети от объёма обучающей выборки при разных стратегиях очистки."""
    df = pd.read_csv(RES / "exp4_sample_size.csv")
    conts = sorted(df.contamination.unique())
    fig, axes = plt.subplots(1, len(conts), figsize=(W, 2.9), sharey=True, squeeze=False)
    axes = axes[0]
    q_full = df.Q.max()
    for i, (ax, c_) in enumerate(zip(axes, conts)):
        d = df[df.contamination == c_]
        for key, label, col, ls, mk in FIG5_SERIES:
            g = d[d.strategy == key]
            rows = [(q, *ci95(v.rmse.values)) for q, v in g.groupby("Q")]
            q, m, lo, hi = map(np.array, zip(*rows))
            ax.plot(q, m, color=col, ls=ls, marker=mk, label=label)
            ax.fill_between(q, lo, hi, color=col, alpha=0.11, lw=0)
        target = d[(d.strategy == "без очистки") & (d.Q == q_full)].rmse.mean()
        ax.axhline(target, color=MUTED, lw=0.8, ls=(0, (4, 2)))
        ax.set_xscale("log")
        ticks = sorted(d.Q.unique())
        major = [t for t in ticks if t in (100, 200, 400, q_full)]
        ax.set_xticks(major)
        ax.set_xticks([t for t in ticks if t not in major], minor=True)
        ax.set_xticklabels([str(t) for t in major])
        ax.tick_params(axis="x", which="minor", length=3)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"({'аб'[i]}) {comma(c_ * 100)} % выбросов", fontsize=9, loc="left", pad=3)
        ax.set_xlabel("объём обучающей выборки $Q$")
        _fmt_axes(ax, x=False)
    axes[0].set_ylabel("RMSE на тесте, МПа")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=9,
               columnspacing=1.2, handlelength=2.4)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.13, 1, 1))
    p = OUT / "ris4_obyom_vyborki.png"
    fig.savefig(p)
    plt.close(fig)
    return p


if __name__ == "__main__":
    for fn in [fig_magnitude, fig_peeling, fig_optimizer, fig_x_outliers, fig_sample_size_rmse]:
        try:
            print(fn.__name__, "->", fn())
        except FileNotFoundError as e:
            print(fn.__name__, "пропущен:", e.filename)
