"""Общие константы и утилиты экспериментов."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from nnfa_core import f1_at_k

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("LAB6_RESULTS", ROOT / "results"))
RESULTS.mkdir(exist_ok=True)

# SMOKE=1 — быстрый прогон всех сценариев с урезанными сетками для проверки кода
SMOKE = os.environ.get("SMOKE") == "1"
N_JOBS = 2

NNF = "NNF (ξ=0)"
NNFA = "NNFA (ξ=0,1)"

# Названия методов для текста, таблиц и рисунков: NNF (Neural Network Filter) и NNFA
# (Neural Network Filter with Averaging) — как в препринте; ключи в таблицах результатов те же.
DISPLAY = {
    NNF: "NNF (ξ = 0)",
    NNFA: "NNFA (ξ = 0,1)",
    "NNFA-peel (автоостановка)": "NNFA, последовательное удаление (автоостановка)",
    "NNFA-порог (один шаг)": "NNFA, один шаг, порог λ(n)",
    "NNFA top-k (оракул)": "NNFA, один шаг, топ-m (m известно)",
    "NNFA top-k (оракул k)": "NNFA, топ-m (m известно)",
}


def pretty_name(name: str) -> str:
    """Название метода для текста, таблиц и рисунков."""
    return DISPLAY.get(name, name)


def evaluate(scores: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """F1 при известном числе выбросов (top-k) и ROC-AUC по скорам."""
    scores = np.asarray(scores, float)
    auc = 0.5 if np.ptp(scores) == 0 else float(roc_auc_score(truth, scores))
    return float(f1_at_k(scores, truth)), auc


def prf(pred_idx, truth: np.ndarray) -> tuple[float, float, float]:
    """Точность, полнота и F1 для множества объявленных выбросов."""
    pred = np.zeros(len(truth), int)
    pred[list(pred_idx)] = 1
    tp = int((pred & truth).sum())
    p = tp / pred.sum() if pred.sum() else 0.0
    r = tp / truth.sum() if truth.sum() else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f
