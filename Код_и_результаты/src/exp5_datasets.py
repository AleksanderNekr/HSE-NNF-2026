"""
Эксперимент 5 — ключевые эксперименты на дополнительных наборах данных.

Наборы: California Housing (подвыборка 1030 районов) и функция Фридмана № 1
(1030 точек). Повторяются с теми же настройками, что для Concrete:
  * зависимость качества обнаружения от мощности выбросов (эксп. 1, 30 повторов);
  * последовательное удаление с автоостановкой (эксп. 2, 15 повторов);
  * польза очистки для итоговой модели (эксп. 3e, 30 разбиений).
Объём выборки для этих наборов посчитан в отдельном ноутбуке
«Очистка_и_объём_выборки».

Результаты: results/exp5_<набор>_magnitude.csv, _peeling.csv, _trajectories.csv, _downstream.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import N_JOBS, RESULTS, SMOKE
from exp1_magnitude import K_VALUES, N_RUNS, run_magnitude
from exp2_peeling import CONTAMINATIONS, MODES, one_run
from exp3_extensions import _run_3e

DATASETS = ["california", "friedman"]
N_PEEL = 2 if SMOKE else 15
N_SPLITS = 2 if SMOKE else 30


def run(ds: str) -> None:
    t0 = time.time()
    out = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(run_magnitude)(k, r, ds)
                                             for k in K_VALUES for r in range(N_RUNS))
    pd.DataFrame([x for o in out for x in o]).to_csv(RESULTS / f"exp5_{ds}_magnitude.csv", index=False)
    print(f"{ds}: мощность готова за {(time.time() - t0) / 60:.1f} мин", flush=True)

    t0 = time.time()
    out = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(one_run)(c, m, r, ds)
                                             for c in CONTAMINATIONS for m in MODES for r in range(N_PEEL))
    pd.DataFrame([x for o, _ in out for x in o]).to_csv(RESULTS / f"exp5_{ds}_peeling.csv", index=False)
    pd.DataFrame([x for _, t in out for x in t]).to_csv(RESULTS / f"exp5_{ds}_trajectories.csv", index=False)
    print(f"{ds}: последовательное удаление готово за {(time.time() - t0) / 60:.1f} мин", flush=True)

    t0 = time.time()
    out = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(_run_3e)(s, ds) for s in range(N_SPLITS))
    pd.DataFrame([x for o in out for x in o]).to_csv(RESULTS / f"exp5_{ds}_downstream.csv", index=False)
    print(f"{ds}: польза очистки готова за {(time.time() - t0) / 60:.1f} мин", flush=True)


if __name__ == "__main__":
    for ds in (sys.argv[1:] or DATASETS):
        run(ds)
