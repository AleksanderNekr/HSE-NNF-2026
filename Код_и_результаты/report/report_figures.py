"""Рисунки итогового отчёта (оформление как в src/paper_figures.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import paper_figures as P  # noqa: E402  (стиль, палитра, ci95, comma)
from paper_figures import C, MUTED, W, ci95, plt, _fmt_axes  # noqa: E402
from report_stats import DS, DS_NAME, _path, SS_RES  # noqa: E402

OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)


def fig_magnitude3() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.35), sharey=True)
    for ax, ds, letter in zip(axes, DS, "абв"):
        df = pd.read_csv(_path(ds, "magnitude"))
        for key, label, col, ls, mk in P.FIG1_SERIES:
            d = df[df.method == key]
            rows = [(k, *ci95(g.f1.values)) for k, g in d.groupby("k_sigma")]
            k, m, lo, hi = map(np.array, zip(*rows))
            ax.plot(k, m, color=col, ls=ls, marker=mk, ms=3, label=label,
                    lw=0.9 if key.startswith("Контроль") else 1.2)
            ax.fill_between(k, lo, hi, color=col, alpha=0.12, lw=0)
        ax.axhline(0.5, color="#999999", lw=0.6, ls=(0, (4, 3)))
        ax.set_title(f"({letter}) {DS_NAME[ds]}", fontsize=9.5, loc="left")
        ax.set_xlabel("$k$, в долях $\\sigma_y$", fontsize=9.5)
        ax.set_xticks([0.5, 1, 1.5, 2, 2.5, 3])
        ax.set_ylim(-0.02, 1.02)
        _fmt_axes(ax)
    axes[0].set_ylabel("F1")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout(w_pad=0.6)
    p = OUT / "ris2_moshchnost_3_nabora.png"
    fig.savefig(p)
    plt.close(fig)
    return p


SS_SERIES = [("без очистки", "без очистки", "#333333", "-", "o"),
             ("NNFA, m известно", "NNFA, $m$ известно", C[0], "-", "s"),
             ("NNFA, порог λ(n)", "NNFA, порог $\\lambda(n)$", C[2], "--", "^"),
             ("последовательное удаление", "последовательное удаление", C[1], "-.", "D"),
             ("идеальная очистка", "идеальная очистка", C[3], ":", "v")]
SS_NAME = {"Concrete": "Concrete, МПа", "California": "California, сотни тыс. долл.", "Friedman": "функция Фридмана"}


def fig_sample_size3(c: float = 0.10) -> Path:
    ex = pd.read_csv(SS_RES / "experiment.csv")
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.35))
    for ax, ds, letter in zip(axes, ["Concrete", "California", "Friedman"], "абв"):
        d0 = ex[(ex.dataset == ds) & (ex.contamination == c)]
        for key, label, col, ls, mk in SS_SERIES:
            d = d0[d0.strategy == key]
            if d.empty:
                continue
            rows = [(q, *ci95(g.rmse.values)) for q, g in d.groupby("Q")]
            q, m, lo, hi = map(np.array, zip(*rows))
            ax.plot(q, m, color=col, ls=ls, marker=mk, ms=3, lw=1.2, label=label)
            ax.fill_between(q, lo, hi, color=col, alpha=0.11, lw=0)
        base = d0[d0.strategy == "без очистки"].groupby("Q").rmse.mean()
        ax.axhline(base.iloc[-1], color="#555555", lw=0.6, ls=(0, (4, 3)))
        ax.set_xscale("log")
        qs = sorted(d0.Q.unique())
        ticks = [qs[0], qs[len(qs) // 2], qs[-1]]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _p: f"{int(round(x))}"))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_title(f"({letter}) {SS_NAME[ds]}", fontsize=9.5, loc="left")
        ax.set_xlabel("объём выборки $Q$", fontsize=9.5)
        _fmt_axes(ax, x=False)
    axes[0].set_ylabel("RMSE на тесте")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8.0, bbox_to_anchor=(0.5, -0.13),
               handlelength=2.2, columnspacing=1.0)
    fig.tight_layout(w_pad=0.8)
    p = OUT / "ris3_obyom_3_nabora.png"
    fig.savefig(p)
    plt.close(fig)
    return p


if __name__ == "__main__":
    for f in (fig_magnitude3, fig_sample_size3):
        try:
            print(f.__name__, "->", f())
        except FileNotFoundError as e:
            print(f.__name__, "пропущен:", e)
