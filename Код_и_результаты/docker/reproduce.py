"""
Полный повтор вычислений в отдельной папке со сверкой с сохранёнными results/.

Код (src/) и данные (data/) копируются в папку повтора (по умолчанию repro_run/),
там заново считаются все эксперименты, строятся рисунки и выполняется verify.py.
Сохранённые results/, figures/ и figures_paper/ проекта не трогаются. В конце
compare_results.py сравнивает новые таблицы с сохранёнными и пишет отчёт
repro_run/СВЕРКА.md.

Запуск (из папки «Код_и_результаты»):
    python docker/reproduce.py              полный повтор, около 4 ч на двух ядрах
    python docker/reproduce.py --smoke      быстрая проверка кода (урезанные сетки, несколько минут)
    python docker/reproduce.py --only exp1_magnitude exp2_peeling
Уже посчитанные в папке повтора шаги пропускаются, поэтому прерванный запуск можно продолжить.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # папка «Код_и_результаты»

# (шаг, итоговый CSV, аргументы) — порядок как в run_all.sh, плюс exp5 и код авторов
STEPS = [
    ("exp3g_optimizer", "exp3g_optimizer.csv", ["exp3g_optimizer.py"]),
    ("exp1_magnitude", "exp1_magnitude.csv", ["exp1_magnitude.py"]),
    ("exp1b_edge", "exp1b_edge.csv", ["exp1b_edge.py"]),
    ("exp3a_x_outliers", "exp3a_x_outliers.csv", ["exp3_extensions.py", "3a"]),
    ("exp3c_n_scan", "exp3c_n_scan.csv", ["exp3_extensions.py", "3c"]),
    ("exp3d_sample_size", "exp3d_sample_size.csv", ["exp3_extensions.py", "3d"]),
    ("exp2_peeling", "exp2_peeling.csv", ["exp2_peeling.py"]),
    ("exp2c_clean", "exp2c_clean.csv", ["exp2c_clean.py"]),
    ("exp3e_downstream", "exp3e_downstream.csv", ["exp3_extensions.py", "3e"]),
    ("exp3b_xi_sensitivity", "exp3b_xi_sensitivity.csv", ["exp3_extensions.py", "3b"]),
    ("exp4_sample_size", "exp4_sample_size.csv", ["exp4_sample_size.py"]),
    ("exp5_california", "exp5_california_downstream.csv", ["exp5_datasets.py", "california"]),
    ("exp5_friedman", "exp5_friedman_downstream.csv", ["exp5_datasets.py", "friedman"]),
    ("exp1c_authors_code", "exp1c_authors_code.csv", ["exp1c_authors_code.py"]),  # нужен torch
]
FINAL = [("paper_figures", ["paper_figures.py"]),
         ("notebook_figures", ["notebook_figures.py"]),
         ("verify", ["verify.py"])]


def log(msg: str, logfile: Path) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with logfile.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def prepare(out: Path) -> None:
    """Свежая копия кода и данных в папке повтора (результаты прошлых шагов остаются)."""
    ign = shutil.ignore_patterns("__pycache__", "*.pyc")
    for d in ["src", "data"]:
        shutil.copytree(ROOT / d, out / d, dirs_exist_ok=True, ignore=ign)
    nb = "!!!_14_Testing_on_the_Concrete_Strength_dataset.ipynb"
    if (ROOT / "original_repo" / nb).exists():
        (out / "original_repo").mkdir(exist_ok=True)
        shutil.copy2(ROOT / "original_repo" / nb, out / "original_repo" / nb)
    for d in ["results", "figures", "figures_paper", "logs"]:
        (out / d).mkdir(exist_ok=True)


def run(name: str, args: list[str], out: Path, env: dict, progress: Path) -> bool:
    t0 = time.time()
    log(f"старт: {name} ({' '.join(args)})", progress)
    with (out / "logs" / f"{name}.log").open("w", encoding="utf-8") as lf:
        p = subprocess.run([sys.executable, "-u", *args], cwd=out / "src", env=env,
                           stdout=lf, stderr=subprocess.STDOUT)
    dt = (time.time() - t0) / 60
    log(f"{'готово' if p.returncode == 0 else 'ОШИБКА (код %d)' % p.returncode}: {name}, {dt:.1f} мин",
        progress)
    return p.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="быстрая проверка с урезанными сетками")
    ap.add_argument("--out", default=None, help="папка повтора (по умолчанию repro_run или repro_smoke)")
    ap.add_argument("--only", nargs="*", default=None, help="выполнить только эти шаги")
    a = ap.parse_args()

    out = Path(a.out) if a.out else ROOT / ("repro_smoke" if a.smoke else "repro_run")
    out.mkdir(parents=True, exist_ok=True)
    prepare(out)
    progress = out / "logs" / "progress.log"

    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               LAB6_RESULTS=str(out / "results"))
    env.pop("SMOKE", None)
    if a.smoke:
        env["SMOKE"] = "1"

    import joblib, numpy, pandas, pyod, scipy, sklearn, matplotlib  # noqa: E401
    vers = (f"Python {platform.python_version()}, numpy {numpy.__version__}, pandas {pandas.__version__}, "
            f"scipy {scipy.__version__}, scikit-learn {sklearn.__version__}, "
            f"matplotlib {matplotlib.__version__}, pyod {pyod.__version__}, joblib {joblib.__version__}")
    has_torch = importlib.util.find_spec("torch") is not None
    if has_torch:
        import torch
        vers += f", torch {torch.__version__}"
    log(f"папка повтора: {out}; {'SMOKE' if a.smoke else 'полный расчёт'}; ядер: {os.cpu_count()}", progress)
    log(vers, progress)

    failed = []
    for name, csv, args in STEPS:
        if a.only and name not in a.only:
            continue
        if name == "exp1c_authors_code" and not has_torch:
            log("пропуск: exp1c_authors_code — не установлен torch", progress)
            continue
        if (out / "results" / csv).exists():
            log(f"пропуск: {name} — {csv} уже посчитан", progress)
            continue
        if not run(name, args, out, env, progress):
            failed.append(name)
    if not a.only:
        for name, args in FINAL:
            if not run(name, args, out, env, progress):
                failed.append(name)
        cmp = [sys.executable, str(HERE / "compare_results.py"), str(ROOT / "results"), str(out / "results"),
               "--report", str(out / "СВЕРКА.md")]
        subprocess.run(cmp, env=env)
    log("ВСЁ ГОТОВО" if not failed else f"ЗАВЕРШЕНО С ОШИБКАМИ: {', '.join(failed)}", progress)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
