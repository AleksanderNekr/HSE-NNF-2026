"""Листинги приложения: сокращённые фрагменты модулей src/ (полные тексты — в папке src)."""

L1 = '''
# src/nnfa_core.py (фрагмент): интервал АКХН и оценка наблюдений методами NNF/NNFA
def akhn_bounds(Q, N_x, N_y=1):
    log2q = math.log2(Q)
    n_min = (N_y * Q) / ((1 + log2q) * (N_x + N_y)) + 1      # как в коде авторов: +1
    n_max = (N_y / (N_x + N_y)) * ((Q / N_x + 1) * (N_x + N_y + 1) + 1) - 1
    if n_max > Q:
        n_max = min(Q // 2, 20)
    return n_min, n_max

def n_range(Q, N_x, N_y=1, xi=0.0):
    n_min, n_max = akhn_bounds(Q, N_x, N_y)
    n_lim = n_min + xi * (n_max - n_min)                     # xi = 0 -> NNF, xi > 0 -> NNFA
    start = max(1, int(np.ceil(n_min)))
    return list(range(start, max(start, int(np.ceil(n_lim))) + 1))

def train_mlp_errors(X, y, n_hidden, epochs=500, lr=0.01, rng=None):
    """Сеть Linear -> Tanh -> Linear, Adam, полный батч; квадраты ошибок обучения."""
    W1, b1 = _torch_linear_init(rng, X.shape[1], n_hidden)
    W2, b2 = _torch_linear_init(rng, n_hidden, 1)
    opt, yv = _Adam(lr=lr), y.reshape(-1, 1)
    for _ in range(epochs):
        a1 = np.tanh(X @ W1 + b1)
        g_out = (2.0 / len(y)) * (a1 @ W2 + b2 - yv)          # производная MSE
        g_z1 = (g_out @ W2.T) * (1.0 - a1 * a1)
        opt.step([W1, b1, W2, b2],
                 [X.T @ g_z1, g_z1.sum(0), a1.T @ g_out, g_out.sum(0)])
    return ((np.tanh(X @ W1 + b1) @ W2 + b2 - yv) ** 2).ravel()

def nnfa_scores(X, y, xi=0.0, seed=0, y_range=None, units="scaled"):
    Xs = scale_minus1_1(X)                                   # входы в [-1; 1]
    lo, hi = (y.min(), y.max()) if y_range is None else y_range
    half = (hi - lo) / 2
    ys = (y - lo) / half - 1.0                               # выход в [-1; 1]
    rng = np.random.default_rng(seed)
    errors = np.array([train_mlp_errors(Xs, ys, N, rng=rng)
                       for N in n_range(len(y), X.shape[1], 1, xi=xi)])
    if units == "original":
        errors = errors * half ** 2                          # квадрат невязки в единицах y
    return errors.mean(axis=0)                               # среднее по ансамблю
'''

L2 = '''
# src/datasets.py (фрагмент): внесение структурных выбросов мощностью k_sigma * std(y)
def inject_y_outliers(y, n_out, k_sigma, rng, mode="isolated", X=None):
    y = y.copy()
    sigma = y.std()
    signs = -np.sign(y - np.median(y))                       # сдвиг к центру распределения
    signs[signs == 0] = 1.0
    y_new = y + signs * k_sigma * sigma
    cand = np.flatnonzero((y_new >= y.min()) & (y_new <= y.max()))   # остаёмся в диапазоне y
    if mode == "isolated":
        idx = rng.choice(cand, size=n_out, replace=False)
    else:                                                    # кластер соседей по входам
        Xs = (X - X.mean(0)) / X.std(0)
        center = int(rng.choice(cand))
        same = cand[signs[cand] == signs[center]]
        idx = same[np.argsort(np.linalg.norm(Xs[same] - Xs[center], axis=1))[:n_out]]
    y[idx] = y_new[idx]
    truth = np.zeros(len(y), dtype=int)
    truth[idx] = 1
    return y, truth
'''

L3 = '''
# src/peeling.py (фрагмент): последовательное удаление с автоматической остановкой
def critical_value(n, alpha=0.05):
    return float(norm.ppf(1.0 - alpha / (2.0 * n)))          # λ(n), поправка Бонферрони

def robust_max_z(e):
    return float(e.max() / (1.4826 * np.median(e)))         # z = max e / (1,4826 med e)

def nnfa_peel(X, y, xi=0.1, alpha=0.05, patience=3, max_frac=0.10, seed=0):
    y_range = (y.min(), y.max())                             # шкала y по исходной выборке
    active = np.arange(len(y))
    order, last_sig, below = [], 0, 0
    for step in range(1, int(max_frac * len(y)) + 1):
        s = nnfa_scores(X[active], y[active], xi=xi, seed=seed + step,
                        y_range=y_range, units="original")
        e = np.sqrt(s)                                       # невязки в единицах y
        z, lam = robust_max_z(e), critical_value(len(e), alpha)
        loc = int(np.argmax(e))
        order.append(int(active[loc]))
        active = np.delete(active, loc)                      # удаляем и переобучаем
        if z >= lam:
            last_sig, below = step, 0
        else:
            below += 1
            if below >= patience:                            # три шага подряд ниже порога
                break
    return order[:last_sig]                                  # объявленные выбросы
'''

L4 = '''
# Очистка_и_объём_выборки.ipynb (фрагмент): один повтор эксперимента с объёмом выборки
def one_job(ds, c, run):
    Xte, yte, Xp, yp = split(ds, run)                        # чистый тест и пул обучения
    rows = []
    for Q in SIZES[ds]:                                      # вложенные выборки объёма Q
        X, y0 = Xp[:Q], yp[:Q]
        m = int(round(c * Q))
        y, truth = inject(y0, m, np.random.default_rng([7, run, Q, int(round(c * 1000))]))
        e = nnfa_residuals(X, y, seed=1000 * run + Q)
        plans = {"без очистки": [], "NNFA, порог λ(n)": threshold_flags(e),
                 "NNFA, m известно": np.argsort(e)[-m:],
                 "идеальная очистка": np.flatnonzero(truth)}
        if ds == "Concrete":
            plans["последовательное удаление"] = peel(X, y, seed=100_000 * (run + 1) + Q)
        for name, removed in plans.items():
            keep = np.ones(Q, bool)
            keep[removed] = False                            # итоговая модель: MLP (32, 16)
            rows.append(dict(dataset=ds, contamination=c, run=run, Q=Q, strategy=name,
                             rmse=final_rmse(X[keep], y[keep], Xte, yte, seed=10 * run)))
    return rows

def gain(dirty, clean, target):
    """Во сколько раз меньше данных нужно после очистки для точности target."""
    qd, exact_d = q_needed(dirty, target)                    # интерполяция по log Q
    qc, exact_c = q_needed(clean, target)
    return qd / qc, not exact_c                              # (кратность, нижняя граница)
'''
