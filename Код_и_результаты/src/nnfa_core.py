"""
Ядро методов NNF / NNFA (Neural Network Filter / with Averaging).
NNF — Neural Network Filter (одна сеть, ξ = 0); NNFA — Neural Network Filter with Averaging (ансамбль, ξ > 0).

Реализация повторяет логику программных кодов из репозитория препринта
Yasnitsky et al. (2026) — https://github.com/Astrohaker/OutliersDetectionResearch —
но перенесена с PyTorch на NumPy. Причина: эксперименты требуют
десятков тысяч обучений крошечных сетей, а для сетей такого размера
накладные расходы фреймворка превышают полезную работу.

Совместимость с оригиналом сохранена по всем существенным деталям:
  * архитектура  Linear(Nx, N) -> Tanh -> Linear(N, 1);
  * инициализация весов как в torch.nn.Linear: U(-1/sqrt(fan_in), 1/sqrt(fan_in));
  * оптимизатор Adam(lr=0.01, betas=(0.9, 0.999), eps=1e-8), full-batch;
  * функция потерь MSE, 500 эпох по умолчанию;
  * масштабирование входов и выхода в [-1, 1];
  * скор наблюдения = квадрат ошибки ОБУЧЕНИЯ (не прогнозирования),
    усреднённый по ансамблю сетей с разным числом скрытых нейронов N.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# 1. Границы Арнольда — Колмогорова — Хехт-Нильсена (АКХН)
# --------------------------------------------------------------------------


def akhn_bounds(Q: int, N_x: int, N_y: int = 1, variant: str = "repo") -> tuple[float, float]:
    """Допустимый диапазон числа нейронов скрытого слоя.

    variant="paper" — формулы (4)-(5) препринта в чистом виде.
    variant="repo"  — та же формула со сдвигом на ±1 и страховочным
                      ограничением сверху, как в коде авторов. Именно этот
                      вариант воспроизводит опубликованные числа, поэтому он
                      взят по умолчанию.
    """
    log2q = math.log2(Q)
    n_min = (N_y * Q) / ((1 + log2q) * (N_x + N_y))
    n_max = (N_y / (N_x + N_y)) * ((Q / N_x + 1) * (N_x + N_y + 1) + 1)
    if variant == "repo":
        n_min += 1
        n_max -= 1
        if n_max > Q:
            # страховка авторов от абсурдно широкого диапазона на малых выборках
            n_max = min(Q // 2, 20)
    elif variant != "paper":
        raise ValueError(f"unknown variant: {variant}")
    return n_min, n_max


def n_range(Q: int, N_x: int, N_y: int = 1, xi: float = 0.0,
            variant: str = "repo") -> list[int]:
    """Список значений N, по которым усредняются ошибки обучения.

    xi = 0 -> единственная сеть с N = ceil(N_min): это метод NNF.
    xi = 1 -> весь интервал АКХН: полный NNFA.
    """
    n_min, n_max = akhn_bounds(Q, N_x, N_y, variant)
    n_lim = n_min + xi * (n_max - n_min)
    start = max(1, int(np.ceil(n_min)))
    end = max(start, int(np.ceil(n_lim)))
    return list(range(start, end + 1))


# --------------------------------------------------------------------------
# 2. Минимальный MLP на NumPy: Linear -> Tanh -> Linear, Adam, full-batch
# --------------------------------------------------------------------------


@dataclass
class _Adam:
    lr: float = 0.01
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self):
        self.m: dict[int, np.ndarray] = {}
        self.v: dict[int, np.ndarray] = {}
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.m:
                self.m[i] = np.zeros_like(p)
                self.v[i] = np.zeros_like(p)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            p -= self.lr * (self.m[i] / bc1) / (np.sqrt(self.v[i] / bc2) + self.eps)


@dataclass
class _Rprop:
    """Rprop в точности как torch.optim.Rprop (etas 0.5/1.2, шаг 1e-6..50).

    Нужен потому, что рисунки «лотереи N» в репозитории препринта посчитаны
    именно этим оптимизатором, а не Adam, которым считается основной бенчмарк.
    """
    lr: float = 0.01
    eta_minus: float = 0.5
    eta_plus: float = 1.2
    step_min: float = 1e-6
    step_max: float = 50.0

    def __post_init__(self):
        self.prev: dict[int, np.ndarray] = {}
        self.step: dict[int, np.ndarray] = {}

    def step_update(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.prev:
                self.prev[i] = np.zeros_like(p)
                self.step[i] = np.full_like(p, self.lr)
            sign = np.sign(g * self.prev[i])
            factor = np.ones_like(p)
            factor[sign > 0] = self.eta_plus
            factor[sign < 0] = self.eta_minus
            self.step[i] = np.clip(self.step[i] * factor, self.step_min, self.step_max)
            g = g.copy()
            g[sign < 0] = 0.0            # torch обнуляет градиент при смене знака
            p -= np.sign(g) * self.step[i]
            self.prev[i] = g


def _torch_linear_init(rng: np.random.Generator, fan_in: int, fan_out: int):
    """Инициализация, совпадающая с torch.nn.Linear по умолчанию."""
    k = 1.0 / math.sqrt(fan_in)
    W = rng.uniform(-k, k, size=(fan_in, fan_out)).astype(np.float64)
    b = rng.uniform(-k, k, size=(fan_out,)).astype(np.float64)
    return W, b


def train_mlp_errors(X: np.ndarray, y: np.ndarray, n_hidden: int,
                     epochs: int = 500, lr: float = 0.01,
                     rng: np.random.Generator | None = None,
                     optimizer: str = "adam") -> np.ndarray:
    """Обучает сеть на ВСЕЙ выборке и возвращает квадрат ошибки обучения
    для каждого наблюдения. Это и есть сигнал, на котором строится NNF.

    optimizer="adam"  — конфигурация основного бенчмарка препринта;
    optimizer="rprop" — конфигурация, которой посчитаны рисунки «лотереи N».
    """
    rng = rng or np.random.default_rng(0)
    n, d = X.shape
    n_hidden = max(int(n_hidden), 1)        # N < 1 трактуется как один нейрон
    W1, b1 = _torch_linear_init(rng, d, n_hidden)
    W2, b2 = _torch_linear_init(rng, n_hidden, 1)
    params = [W1, b1, W2, b2]
    if optimizer == "adam":
        opt = _Adam(lr=lr)
        step = opt.step
    elif optimizer == "rprop":
        opt = _Rprop(lr=lr)
        step = opt.step_update
    else:
        raise ValueError(optimizer)
    yv = y.reshape(-1, 1)

    for _ in range(epochs):
        z1 = X @ W1 + b1
        a1 = np.tanh(z1)
        out = a1 @ W2 + b2
        diff = out - yv
        # d(MSE)/d(out), MSE усредняется по всем элементам как в torch.nn.MSELoss
        g_out = (2.0 / n) * diff
        gW2 = a1.T @ g_out
        gb2 = g_out.sum(axis=0)
        g_a1 = g_out @ W2.T
        g_z1 = g_a1 * (1.0 - a1 * a1)
        gW1 = X.T @ g_z1
        gb1 = g_z1.sum(axis=0)
        step(params, [gW1, gb1, gW2, gb2])

    out = np.tanh(X @ W1 + b1) @ W2 + b2
    return ((out - yv) ** 2).ravel()


# --------------------------------------------------------------------------
# 3. Масштабирование в [-1, 1] — как в коде авторов
# --------------------------------------------------------------------------


def scale_minus1_1(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    lo = a.min(axis=0)
    hi = a.max(axis=0)
    rng_ = hi - lo
    rng_ = np.where(rng_ == 0, 1.0, rng_)
    return 2.0 * (a - lo) / rng_ - 1.0


# --------------------------------------------------------------------------
# 4. Сами методы
# --------------------------------------------------------------------------


def nnfa_scores(X: np.ndarray, y: np.ndarray, xi: float = 0.0,
                epochs: int = 500, seed: int = 0, lr: float = 0.01,
                variant: str = "repo",
                return_matrix: bool = False,
                y_range: tuple[float, float] | None = None,
                units: str = "scaled"):
    """Оценка «подозрительности» каждого наблюдения методом NNF/NNFA.

    Возвращает вектор усреднённых по ансамблю квадратов ошибки обучения.
    При xi=0 ансамбль вырождается в одну сеть, то есть в метод NNF.

    y_range — фиксированные границы масштабирования y. По умолчанию берутся
    min/max текущей выборки, как у авторов. При последовательном удалении
    границы фиксируются по исходной выборке: иначе каждое удаление крайнего
    наблюдения меняет шкалу и все ошибки «растут» без всякой причины.
    units="original" — ошибки в исходных единицах y (квадрат невязки).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    Q, N_x = X.shape
    Xs = scale_minus1_1(X)
    lo, hi = (y.min(), y.max()) if y_range is None else y_range
    half = (hi - lo) / 2 if hi > lo else 1.0
    ys = (y - lo) / half - 1.0

    Ns = n_range(Q, N_x, 1, xi=xi, variant=variant)
    rng = np.random.default_rng(seed)
    errors = np.empty((len(Ns), Q), dtype=np.float64)
    for i, N in enumerate(Ns):
        errors[i] = train_mlp_errors(Xs, ys, N, epochs=epochs, lr=lr, rng=rng)

    if units == "original":
        errors = errors * half ** 2
    elif units != "scaled":
        raise ValueError(units)
    scores = errors.mean(axis=0)
    if return_matrix:
        return scores, errors, Ns
    return scores


def nnf_scores(X, y, epochs: int = 500, seed: int = 0, **kw):
    """NNF — одна сеть с минимальным допустимым N (частный случай NNFA, xi=0)."""
    return nnfa_scores(X, y, xi=0.0, epochs=epochs, seed=seed, **kw)


# --------------------------------------------------------------------------
# 5. Вспомогательное: top-k и метрики
# --------------------------------------------------------------------------


def top_k_flags(scores: np.ndarray, k: int) -> np.ndarray:
    pred = np.zeros(len(scores), dtype=int)
    if k > 0:
        pred[np.argsort(scores)[-k:]] = 1
    return pred


def f1_at_k(scores: np.ndarray, truth: np.ndarray, k: int | None = None) -> float:
    k = int(truth.sum()) if k is None else k
    pred = top_k_flags(scores, k)
    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0
