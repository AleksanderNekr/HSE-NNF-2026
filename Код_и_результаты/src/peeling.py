"""
Последовательное удаление выбросов («снятие слоёв луковицы») на основе NNFA
с автоматическим критерием остановки.

Алгоритм NNFA-peel
------------------
1. Обучить ансамбль NNFA на текущем наборе и получить для каждого наблюдения
   среднеквадратичную невязку обучения e_i в исходных единицах y.
2. Вычислить робастную статистику максимальной невязки
       z = max_i e_i / (1.4826 · median_i e_i).
   Для |невязок| нормального шума 1.4826·медиана — состоятельная оценка σ,
   а медиана устойчива к ещё не удалённым выбросам.
3. Сравнить z с критическим значением λ(n) = Φ⁻¹(1 − α / (2n)) — это
   двусторонний порог нормального распределения с поправкой Бонферрони на
   n одновременных проверок. Порог теоретический, НЕ подбирается по данным.
4. Удалить наблюдение с максимальной невязкой и перейти к шагу 1.
5. Остановиться, когда z < λ(n) выполняется `patience` шагов подряд.
   Выбросами объявляются все наблюдения, удалённые до последнего шага, на
   котором z ≥ λ(n).

Шаги 2–5 — это обобщённый ESD-тест Рознера (Rosner, 1983), применённый не к
исходным значениям, а к невязкам нейросети, то есть к нарушениям
закономерности. Шкала масштабирования y фиксируется по исходной выборке,
чтобы удаление крайних наблюдений не меняло масштаб ошибок.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from nnfa_core import nnfa_scores


def critical_value(n: int, alpha: float = 0.05) -> float:
    """Двусторонний порог нормального закона с поправкой Бонферрони."""
    return float(norm.ppf(1.0 - alpha / (2.0 * n)))


def robust_max_z(e: np.ndarray) -> float:
    """Отношение максимальной невязки к робастной оценке σ."""
    med = float(np.median(e))
    return float(e.max() / (1.4826 * med)) if med > 0 else float("inf")


@dataclass
class PeelResult:
    detected: list[int]                     # глобальные индексы выбросов
    order: list[int] = field(default_factory=list)   # порядок удаления (все шаги)
    z: list[float] = field(default_factory=list)     # статистика на каждом шаге
    lam: list[float] = field(default_factory=list)   # порог на каждом шаге
    n_steps: int = 0


def nnfa_peel(X: np.ndarray, y: np.ndarray, xi: float = 0.1, alpha: float = 0.05,
              patience: int = 3, max_frac: float = 0.10, seed: int = 0,
              min_steps: int = 0) -> PeelResult:
    """Последовательное удаление с онлайн-критерием остановки.

    min_steps — сколько шагов сделать в любом случае (только чтобы собрать
    траекторию для иллюстраций; на решение о выбросах не влияет).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    y_range = (float(y.min()), float(y.max()))
    active = np.arange(len(y))
    res = PeelResult(detected=[])
    last_sig = 0
    below = 0
    stopped = False          # онлайн-решение принято; дальнейшие шаги — только для траектории
    max_steps = int(max_frac * len(y))

    for step in range(1, max_steps + 1):
        s = nnfa_scores(X[active], y[active], xi=xi, seed=seed + step,
                        y_range=y_range, units="original")
        e = np.sqrt(s)
        z = robust_max_z(e)
        lam = critical_value(len(e), alpha)
        loc = int(np.argmax(e))

        res.order.append(int(active[loc]))
        res.z.append(z)
        res.lam.append(lam)
        res.n_steps = step

        if not stopped:
            if z >= lam:
                last_sig = step
                below = 0
            else:
                below += 1
            stopped = below >= patience
        active = np.delete(active, loc)

        if stopped and step >= min_steps:
            break

    res.detected = res.order[:last_sig]
    return res


def decide_online(z: list[float], lam: list[float], patience: int = 3) -> int:
    """Число объявленных выбросов по траектории статистики — ровно то же
    решение, что принимает nnfa_peel в момент онлайн-остановки."""
    last_sig, below = 0, 0
    for step, (zi, li) in enumerate(zip(z, lam), start=1):
        if zi >= li:
            last_sig, below = step, 0
        else:
            below += 1
        if below >= patience:
            break
    return last_sig


def nnfa_oneshot_threshold(X: np.ndarray, y: np.ndarray, xi: float = 0.1,
                           alpha: float = 0.05, seed: int = 0) -> np.ndarray:
    """Одношаговый вариант с тем же статистическим порогом: выбросами
    объявляются все наблюдения, у которых e_i / (1.4826·median e) ≥ λ(n).
    Нужен как честный эталон для оценки пользы именно последовательности
    удалений — без оракульного знания числа выбросов."""
    s = nnfa_scores(X, y, xi=xi, seed=seed, units="original")
    e = np.sqrt(s)
    z = e / (1.4826 * np.median(e))
    return (z >= critical_value(len(e), alpha)).astype(int)
