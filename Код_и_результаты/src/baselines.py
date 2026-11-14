"""
Классические методы обнаружения выбросов: те же пятнадцать, что в препринте,
плюс лес с out-of-bag невязками и контрольный скорер без модели.

Восемь методов берутся из PyOD и работают только с матрицей признаков X.
Остальные так или иначе используют целевую переменную y: модельные ошибки
(Random Forest, MLP, автоэнкодер, их комбинация) и три метода, которым y
подаётся как дополнительный признак.

Отличия от кода авторов, сделанные ради честности сравнения: все признаки
(включая y, когда он подаётся как признак) стандартизуются; MLP обучается на
стандартизованных X; добавлен лес с невязками по out-of-bag прогнозу.

Каждая функция возвращает вектор скоров: чем больше, тем «подозрительнее».
"""

from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest as SkIForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from pyod.models import abod, copod, hbos, iforest, knn, lof, ocsvm, pca

from nnfa_core import _Adam, _torch_linear_init

# --------------------------------------------------------------------------
# Автоэнкодер: та же архитектура, что в коде авторов (16 -> 4 -> 16, ReLU),
# перенесённая на NumPy ради скорости.
# --------------------------------------------------------------------------


def _autoencoder_errors(Xs: np.ndarray, seed: int, epochs: int = 300,
                        lr: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, d = Xs.shape
    W1, b1 = _torch_linear_init(rng, d, 16)
    W2, b2 = _torch_linear_init(rng, 16, 4)
    W3, b3 = _torch_linear_init(rng, 4, 16)
    W4, b4 = _torch_linear_init(rng, 16, d)
    params = [W1, b1, W2, b2, W3, b3, W4, b4]
    opt = _Adam(lr=lr)

    for _ in range(epochs):
        z1 = Xs @ W1 + b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ W2 + b2                      # узкое горло, без активации
        z3 = z2 @ W3 + b3
        a3 = np.maximum(z3, 0.0)
        out = a3 @ W4 + b4
        diff = out - Xs
        g_out = (2.0 / (n * d)) * diff
        gW4, gb4 = a3.T @ g_out, g_out.sum(0)
        g_a3 = g_out @ W4.T
        g_z3 = g_a3 * (z3 > 0)
        gW3, gb3 = z2.T @ g_z3, g_z3.sum(0)
        g_z2 = g_z3 @ W3.T
        gW2, gb2 = a1.T @ g_z2, g_z2.sum(0)
        g_a1 = g_z2 @ W2.T
        g_z1 = g_a1 * (z1 > 0)
        gW1, gb1 = Xs.T @ g_z1, g_z1.sum(0)
        opt.step(params, [gW1, gb1, gW2, gb2, gW3, gb3, gW4, gb4])

    a1 = np.maximum(Xs @ W1 + b1, 0.0)
    z2 = a1 @ W2 + b2
    a3 = np.maximum(z2 @ W3 + b3, 0.0)
    out = a3 @ W4 + b4
    return ((out - Xs) ** 2).mean(axis=1)


# --------------------------------------------------------------------------
# Полный набор классических методов
# --------------------------------------------------------------------------

# Контрольный «метод» без модели: удалённость y от медианы. Если внесённые
# выбросы действительно структурные, он должен работать на уровне случайного
# угадывания — это проверка корректности протокола, а не конкурент.
CONTROL = "Контроль: |y − медиана|"

PYOD_ONLY = [
    "ABOD (pyod)", "HBOS (pyod)", "IsolationForest (pyod)", "kNN (pyod)",
    "LOF (pyod)", "OCSVM (pyod)", "PCA (pyod)", "COPOD (pyod)",
]
WITH_TARGET = [
    "Random Forest (error)", "Neural Network (error)",
    "Autoencoder (reconstruction)", "Combined (RF+AE)",
    "One-Class SVM (with y)", "Isolation Forest (with y)", "LOF (with y)",
]
# Честный вариант леса: невязка по out-of-bag прогнозу, а не на тех же
# точках, на которых дерево обучалось (иначе лес «запоминает» выброс).
RF_OOB = "Random Forest (OOB error)"
CLASSICAL_METHODS = PYOD_ONLY + WITH_TARGET + [RF_OOB]

# ABOD квадратичен по числу наблюдений — на больших выборках его можно отключить
HEAVY = {"ABOD (pyod)"}


def classical_scores(X: np.ndarray, y: np.ndarray, seed: int = 0,
                     methods: list[str] | None = None) -> dict[str, np.ndarray]:
    """Считает скоры всех (или выбранных) классических методов."""
    methods = methods or CLASSICAL_METHODS
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    Xs = StandardScaler().fit_transform(X)
    # y как признак стандартизуется вместе с X: без этого расстояния в LOF и
    # One-Class SVM определяются признаками масштаба сотен, и y почти не
    # влияет на результат
    Xy = StandardScaler().fit_transform(np.column_stack([X, y]))
    out: dict[str, np.ndarray] = {}

    pyod_models = {
        "ABOD (pyod)": lambda: abod.ABOD(),
        "HBOS (pyod)": lambda: hbos.HBOS(),
        "IsolationForest (pyod)": lambda: iforest.IForest(random_state=seed),
        "kNN (pyod)": lambda: knn.KNN(),
        "LOF (pyod)": lambda: lof.LOF(),
        "OCSVM (pyod)": lambda: ocsvm.OCSVM(),
        "PCA (pyod)": lambda: pca.PCA(),
        "COPOD (pyod)": lambda: copod.COPOD(),
    }
    for name, factory in pyod_models.items():
        if name in methods:
            m = factory()
            m.fit(Xs)
            out[name] = np.asarray(m.decision_scores_, float)

    rf_err = ae_err = None
    if {"Random Forest (error)", "Combined (RF+AE)", RF_OOB} & set(methods):
        rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1,
                                   oob_score=True)
        rf.fit(X, y)
        rf_err = (rf.predict(X) - y) ** 2
        if "Random Forest (error)" in methods:
            out["Random Forest (error)"] = rf_err
        if RF_OOB in methods:
            out[RF_OOB] = (rf.oob_prediction_ - y) ** 2

    if CONTROL in methods:
        out[CONTROL] = np.abs(y - np.median(y))

    if "Neural Network (error)" in methods:
        mlp = MLPRegressor(hidden_layer_sizes=(20, 10), random_state=seed, max_iter=500)
        mlp.fit(Xs, y)
        out["Neural Network (error)"] = (mlp.predict(Xs) - y) ** 2

    if {"Autoencoder (reconstruction)", "Combined (RF+AE)"} & set(methods):
        ae_err = _autoencoder_errors(Xs, seed=seed)
        if "Autoencoder (reconstruction)" in methods:
            out["Autoencoder (reconstruction)"] = ae_err

    if "Combined (RF+AE)" in methods:
        out["Combined (RF+AE)"] = (rf_err / rf_err.max() + ae_err / ae_err.max()) / 2

    if "One-Class SVM (with y)" in methods:
        m = OneClassSVM(kernel="rbf", gamma="auto").fit(Xy)
        out["One-Class SVM (with y)"] = -m.decision_function(Xy)

    if "Isolation Forest (with y)" in methods:
        m = SkIForest(random_state=seed).fit(Xy)
        out["Isolation Forest (with y)"] = -m.decision_function(Xy)

    if "LOF (with y)" in methods:
        m = LocalOutlierFactor(novelty=True).fit(Xy)
        out["LOF (with y)"] = -m.score_samples(Xy)

    for k, v in out.items():
        v = np.asarray(v, float)
        finite = np.isfinite(v)
        if not finite.all():
            # бесконечность — максимальная подозрительность, NaN — минимальная
            top = v[finite].max() if finite.any() else 0.0
            bot = v[finite].min() if finite.any() else 0.0
            v = np.where(np.isposinf(v), top + 1.0, v)
            v = np.where(np.isnan(v) | np.isneginf(v), bot - 1.0, v)
        out[k] = v
    return out
