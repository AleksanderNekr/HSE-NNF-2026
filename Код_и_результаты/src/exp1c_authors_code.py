"""
Эксперимент 1c — код авторов (PyTorch) на протоколе препринта, 30 повторов.

Функция run_nnfa_original берётся без изменений из ноутбука авторов
!!!_14_Testing_on_the_Concrete_Strength_dataset.ipynb. Сам ноутбук в этот
репозиторий не входит: его нужно скачать из репозитория авторов
https://github.com/Astrohaker/OutliersDetectionResearch и положить в папку
original_repo/ рядом с src/.
Подготовка данных повторяет их create_concrete_dataset: сортировка по прочности,
10 наибольших и 10 наименьших значений заменяются на (max + min)/2, строки
перемешиваются с seed = номер повтора. Отличие одно: данные читаются из локального
файла, а не с сайта UCI. Нужен PyTorch.

Результат: results/exp1c_authors_code.csv
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS
from datasets import load_concrete

NB = Path(__file__).resolve().parent.parent / "original_repo" / "!!!_14_Testing_on_the_Concrete_Strength_dataset.ipynb"
COLS = ["Cement", "BlastFurnaceSlag", "FlyAsh", "Water", "Superplasticizer",
        "CoarseAggregate", "FineAggregate", "Age"]


def authors_functions() -> dict:
    """Исполняет фрагмент ноутбука авторов с get_top_k_outliers и run_nnfa_original."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    if not NB.is_file():
        raise FileNotFoundError(
            f"Не найден ноутбук авторов {NB.name}: скачайте его из "
            "https://github.com/Astrohaker/OutliersDetectionResearch "
            f"и положите в {NB.parent}")
    src = "".join(json.loads(NB.read_text(encoding="utf-8"))["cells"][0]["source"])
    code = src[src.index("def get_top_k_outliers"):src.index("# 4. Wrappers")]
    ns = dict(np=np, math=math, random=random, torch=torch, nn=nn, optim=optim)
    exec(code, ns)
    return ns


def create_concrete_dataset(seed_shuffle: int):
    """Та же подготовка данных, что у авторов (create_concrete_dataset в !!!_14)."""
    X0, y0 = load_concrete()
    df = pd.DataFrame(X0, columns=COLS)
    df["CompressiveStrength"] = y0
    s_avg = (df.CompressiveStrength.max() + df.CompressiveStrength.min()) / 2
    d = df.sort_values("CompressiveStrength", ascending=False).reset_index(drop=True)
    idx = list(range(10)) + list(range(len(d) - 10, len(d)))
    d.loc[idx, "CompressiveStrength"] = s_avg
    d["is_outlier"] = 0
    d.loc[idx, "is_outlier"] = 1
    d = d.sample(frac=1, random_state=seed_shuffle).reset_index(drop=True)
    X = d.drop(["CompressiveStrength", "is_outlier"], axis=1).values.astype(np.float32)
    y = d.CompressiveStrength.values.astype(np.float32).reshape(-1, 1)
    return X, y, d.is_outlier.values.astype(int)


def main(n_runs: int = 30) -> pd.DataFrame:
    ns = authors_functions()
    rows = []
    for seed in range(n_runs):
        X, y, truth = create_concrete_dataset(seed)
        pred, scores = ns["run_nnfa_original"](X, y, 20, ksi=0, random_state=seed)
        tp = int(((pred == 1) & (truth == 1)).sum())
        fp = int(((pred == 1) & (truth == 0)).sum())
        fn = int(((pred == 0) & (truth == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        rows.append(dict(run=seed, tp=tp, fp=fp, fn=fn,
                         f1=2 * p * r / (p + r) if p + r else 0.0,
                         auc=float(roc_auc_score(truth, scores))))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "exp1c_authors_code.csv", index=False)
    return df


if __name__ == "__main__":
    print(main().describe().round(3))
