"""
Сверка таблиц повтора с сохранёнными таблицами results/.

    python docker/compare_results.py <сохранённые results> <results повтора> [--report СВЕРКА.md]

Столбцы со временем счёта (seconds, clean_seconds, fit_seconds) не сравниваются: они зависят
от компьютера. Остальные столбцы сравниваются построчно. Итог по файлу:
  совпадает         — все значения равны (разница не больше 1e-9);
  почти совпадает   — разница есть, но не больше 1e-6 (порядок суммирования в BLAS);
  расходится        — есть заметные расхождения, приводятся средние метрик до и после.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TIME_COLS = {"seconds", "clean_seconds", "fit_seconds"}
METRICS = ["f1", "auc", "recall", "precision", "rmse", "hit_top1", "n_detected", "n_removed"]
GROUP = ["method", "strategy", "config"]


def compare(saved: Path, new: Path) -> tuple[str, str]:
    a, b = pd.read_csv(saved), pd.read_csv(new)
    if list(a.columns) != list(b.columns):
        return "расходится", f"разные столбцы: {list(a.columns)} и {list(b.columns)}"
    if len(a) != len(b):
        return "расходится", f"разное число строк: {len(a)} и {len(b)}"
    cols = [c for c in a.columns if c not in TIME_COLS]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
    txt = [c for c in cols if c not in num]
    bad_txt = sum(int((a[c].astype(str) != b[c].astype(str)).sum()) for c in txt)
    diffs = {c: float(np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float))))
             if len(a) else 0.0 for c in num}
    nan_mismatch = sum(int((a[c].isna() != b[c].isna()).sum()) for c in num)
    mx = max(diffs.values(), default=0.0)
    if np.isnan(mx):
        mx = 0.0
    rows_diff = 0
    if num:
        d = np.abs(a[num].to_numpy(float) - b[num].to_numpy(float))
        rows_diff = int((np.nan_to_num(d) > 1e-9).any(axis=1).sum())
    detail = f"строк {len(a)}, макс. |Δ| = {mx:.2e}, строк с отличиями: {rows_diff}"
    if bad_txt or nan_mismatch:
        detail += f", отличий в текстовых полях: {bad_txt}, в пропусках: {nan_mismatch}"
        return "расходится", detail
    if mx <= 1e-9:
        return "совпадает", detail
    if mx <= 1e-6:
        return "почти совпадает", detail
    g = next((c for c in GROUP if c in a.columns), None)
    m = [c for c in METRICS if c in num]
    if m:
        sa = a.groupby(g)[m].mean() if g else a[m].mean().to_frame().T
        sb = b.groupby(g)[m].mean() if g else b[m].mean().to_frame().T
        t = pd.concat({"сохранено": sa, "повтор": sb}, axis=1).round(4)
        try:
            detail += "\n\n" + t.to_markdown()
        except ImportError:  # нет пакета tabulate
            detail += "\n\n```\n" + t.to_string() + "\n```"
    return "расходится", detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("saved")
    ap.add_argument("new")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    saved, new = Path(a.saved), Path(a.new)

    lines = ["# Сверка повтора с сохранёнными результатами", "",
             f"Сохранённые таблицы: `{saved}`  ", f"Повтор: `{new}`", "",
             "| Файл | Итог | Подробности |", "|---|---|---|"]
    long = []
    counts: dict[str, int] = {}
    for f in sorted(saved.glob("*.csv")):
        g = new / f.name
        if not g.exists():
            verdict, detail = "нет в повторе", ""
        else:
            verdict, detail = compare(f, g)
        counts[verdict] = counts.get(verdict, 0) + 1
        head, _, tail = detail.partition("\n\n")
        lines.append(f"| {f.name} | {verdict} | {head} |")
        if tail:
            long += [f"## {f.name}", "", "Средние по методам (сохранено / повтор):", "", tail, ""]
        print(f"{verdict:16s} {f.name}: {head}")
    lines += ["", "Итого: " + ", ".join(f"{k} — {v}" for k, v in counts.items()), ""] + long
    if a.report:
        Path(a.report).write_text("\n".join(lines), encoding="utf-8")
        print(f"отчёт: {a.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
