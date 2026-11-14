"""
Наборы данных и управляемое внесение структурных выбросов.

Ключевое отличие от кода препринта: там выброс создавался единственным
способом — подменой целевого значения на среднее по выборке, что жёстко
задаёт его «мощность». Здесь мощность вынесена в параметр k и измеряется
в стандартных отклонениях целевой переменной, что и позволяет построить
кривую обнаружимости.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


# --------------------------------------------------------------------------
# Реальные данные
# --------------------------------------------------------------------------


def load_concrete() -> tuple[np.ndarray, np.ndarray]:
    """Concrete Compressive Strength (Yeh, 1998): 1030 наблюдений, 8 признаков."""
    df = pd.read_csv(DATA / "concrete.csv")
    X = df.iloc[:, :8].values.astype(float)
    y = df["strength"].values.astype(float)
    return X, y


def _make_california(path: Path) -> None:
    """California Housing: признаки как в sklearn.datasets.fetch_california_housing,
    целевая переменная в сотнях тысяч долларов. Признаки с длинными хвостами
    (комнаты, спальни, население, заполненность) логарифмируются, все признаки
    обрезаются по 0,5 %- и 99,5 %-квантилям. Для сопоставимости с Concrete берётся
    случайная подвыборка из 1030 районов (фиксированное зерно)."""
    d = pd.read_csv(DATA / "housing.csv").dropna(subset=["total_bedrooms"])
    hh = d["households"]
    X = np.column_stack([d["median_income"], d["housing_median_age"], d["total_rooms"] / hh,
                         d["total_bedrooms"] / hh, d["population"], d["population"] / hh,
                         d["latitude"], d["longitude"]]).astype(float)
    X[:, 2:6] = np.log(X[:, 2:6])
    X = np.clip(X, np.percentile(X, 0.5, axis=0), np.percentile(X, 99.5, axis=0))
    y = (d["median_house_value"] / 1e5).to_numpy(float)
    idx = np.sort(np.random.default_rng(2026).choice(len(y), 1030, replace=False))
    cols = ["MedInc", "HouseAge", "logAveRooms", "logAveBedrms", "logPopulation",
            "logAveOccup", "Latitude", "Longitude"]
    out = pd.DataFrame(X[idx], columns=cols)
    out["value"] = y[idx]
    out.to_csv(path, index=False)


def _make_friedman(path: Path) -> None:
    """Функция Фридмана № 1: 10 входов, значимы первые пять, шум N(0, 1); 1030 точек."""
    from sklearn.datasets import make_friedman1
    X, y = make_friedman1(n_samples=1030, n_features=10, noise=1.0, random_state=2026)
    out = pd.DataFrame(X, columns=[f"x{i + 1}" for i in range(10)])
    out["y"] = y
    out.to_csv(path, index=False)


def load_dataset(name: str = "concrete") -> tuple[np.ndarray, np.ndarray]:
    """Наборы для экспериментов: concrete, california (подвыборка 1030), friedman (1030)."""
    if name == "concrete":
        return load_concrete()
    path = DATA / f"{name}_1030.csv"
    if not path.exists():
        {"california": _make_california, "friedman": _make_friedman}[name](path)
    d = pd.read_csv(path)
    return d.iloc[:, :-1].to_numpy(float), d.iloc[:, -1].to_numpy(float)


# --------------------------------------------------------------------------
# Синтетика из препринта
# --------------------------------------------------------------------------


def make_parabola(Q: int = 20, noise: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.linspace(1, Q, Q)
    y = (x - 7.0) ** 2
    if noise:
        y = y + rng.normal(0, noise * y.std(), Q)
    return x.reshape(-1, 1), y


def make_paraboloid(Q: int = 45, noise: float = 0.0, seed: int = 0):
    """Параболоид вращения y = x1^2 + x2^2 на регулярной сетке."""
    side = int(np.ceil(np.sqrt(Q)))
    g = np.linspace(-3, 3, side)
    xx, yy = np.meshgrid(g, g)
    X = np.column_stack([xx.ravel(), yy.ravel()])[:Q]
    y = X[:, 0] ** 2 + X[:, 1] ** 2
    if noise:
        y = y + np.random.default_rng(seed).normal(0, noise * y.std(), len(y))
    return X, y


def make_hyperbolic_paraboloid(Q: int = 45, noise: float = 0.0, seed: int = 0):
    """Гиперболический параболоид y = x1^2 - x2^2 («седло»)."""
    side = int(np.ceil(np.sqrt(Q)))
    g = np.linspace(-3, 3, side)
    xx, yy = np.meshgrid(g, g)
    X = np.column_stack([xx.ravel(), yy.ravel()])[:Q]
    y = X[:, 0] ** 2 - X[:, 1] ** 2
    if noise:
        y = y + np.random.default_rng(seed).normal(0, noise * y.std(), len(y))
    return X, y


def make_friedman(Q: int = 400, n_features: int = 10, noise: float = 0.1, seed: int = 0):
    """Задача Фридмана №1: нелинейная зависимость плюс лишние признаки.
    Нужна для проверки поведения методов в более высокой размерности."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(Q, n_features))
    y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2
         + 10 * X[:, 3] + 5 * X[:, 4])
    if noise:
        y = y + rng.normal(0, noise * y.std(), Q)
    return X, y


# --------------------------------------------------------------------------
# Внесение выбросов
# --------------------------------------------------------------------------


def _structural_shift(y: np.ndarray, k_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Направление и допустимость сдвига для каждого наблюдения.

    Сдвиг делается К ЦЕНТРУ распределения (как в препринте, где выброс
    получает среднее значение), а наблюдение допускается к порче, только
    если новое значение остаётся внутри наблюдаемого диапазона y. Так выброс
    гарантированно не экстремален по y и остаётся структурным: он нарушает
    зависимость y(X), но не выделяется ни по одному признаку в отдельности.
    """
    sigma = y.std()
    signs = -np.sign(y - np.median(y))
    signs[signs == 0] = 1.0
    y_new = y + signs * k_sigma * sigma
    feasible = (y_new >= y.min()) & (y_new <= y.max())
    return signs, feasible


def inject_y_outliers(y: np.ndarray, n_out: int, k_sigma: float,
                      rng: np.random.Generator, mode: str = "isolated",
                      X: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Вносит n_out структурных выбросов мощностью k_sigma * std(y).

    mode="isolated" — позиции выбираются равномерно среди допустимых;
    mode="cluster"  — группа из n_out ближайших соседей в стандартизованном
                      пространстве признаков X со сдвигом в одну сторону:
                      соседние аномалии поддерживают друг друга (маскировка).
    """
    y = np.asarray(y, float).copy()
    Q = len(y)
    sigma = y.std()
    signs, feasible = _structural_shift(y, k_sigma)
    cand = np.flatnonzero(feasible)
    if len(cand) < n_out:
        raise ValueError(f"мощность {k_sigma}σ: допустимых наблюдений {len(cand)} < {n_out}")

    if mode == "isolated":
        idx = rng.choice(cand, size=n_out, replace=False)
    elif mode == "cluster":
        if X is None:
            raise ValueError("для mode='cluster' нужна матрица X")
        Xs = (X - X.mean(0)) / X.std(0)
        center = int(rng.choice(cand))
        same_side = cand[signs[cand] == signs[center]]
        d = np.linalg.norm(Xs[same_side] - Xs[center], axis=1)
        idx = same_side[np.argsort(d)[:n_out]]
    else:
        raise ValueError(mode)

    y[idx] = y[idx] + signs[idx] * k_sigma * sigma
    truth = np.zeros(Q, dtype=int)
    truth[idx] = 1
    return y, truth


def inject_preprint_scenario(y: np.ndarray, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """Протокол препринта (ноутбук !!!_15): n_out/2 наибольших и n_out/2
    наименьших значений y заменяются на (max + min) / 2. Точка привязки для
    сверки с опубликованными числами."""
    y = np.asarray(y, float).copy()
    order = np.argsort(y)
    half = n_out // 2
    idx = np.concatenate([order[:half], order[-half:]])
    y[idx] = (y.max() + y.min()) / 2
    truth = np.zeros(len(y), dtype=int)
    truth[idx] = 1
    return y, truth


def inject_x_outliers(X: np.ndarray, n_out: int, k_sigma: float,
                      rng: np.random.Generator,
                      n_features_hit: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Портит значения ВХОДНЫХ признаков — случай, который в препринте
    прямо указан как непроверенный."""
    X = np.asarray(X, float).copy()
    Q, d = X.shape
    idx = rng.choice(Q, size=n_out, replace=False)
    for i in idx:
        cols = rng.choice(d, size=min(n_features_hit, d), replace=False)
        shift = k_sigma * X[:, cols].std(axis=0)
        signs = rng.choice([-1.0, 1.0], size=len(cols))
        # физически невозможные отрицательные дозировки и возраст не допускаем
        signs = np.where(X[i, cols] - shift < 0, 1.0, signs)
        X[i, cols] = X[i, cols] + signs * shift
    truth = np.zeros(Q, dtype=int)
    truth[idx] = 1
    return X, truth
