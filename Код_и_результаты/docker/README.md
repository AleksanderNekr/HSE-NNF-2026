# Повтор расчётов в изолированном окружении

Здесь лежит всё, чтобы пересчитать исследование с нуля на тех же версиях библиотек, на которых
получены таблицы `results/`, и автоматически сверить новые числа со старыми.

| Файл | Назначение |
|---|---|
| `requirements-lock.txt` | точные версии всех пакетов (Python 3.11, numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, scikit-learn 1.8.0, matplotlib 3.10.9, pyod 3.6.6, joblib 1.5.3) |
| `Dockerfile`, `../docker-compose.yml` | образ с этим окружением; код и данные подключаются как папка, в образ не копируются |
| `environment-lock.yml` | то же окружение для conda (вариант без Docker) |
| `reproduce.py` | полный повтор: все эксперименты (exp1–exp5 и код авторов exp1c), рисунки, `verify.py`, сверка |
| `compare_results.py` | сверка таблиц повтора с `results/` и отчёт `СВЕРКА.md` |

Повтор идёт в отдельной папке `repro_run/` (туда копируются `src/` и `data/`), поэтому сохранённые
`results/`, `figures/` и `figures_paper/` не перезаписываются. Прерванный запуск можно продолжить:
уже посчитанные шаги пропускаются. Ход расчёта — в `repro_run/logs/progress.log`, вывод каждого шага —
в `repro_run/logs/<шаг>.log`.

## Вариант 1. Docker (Windows, macOS, Linux)

Нужен Docker Desktop (Windows: `winget install -e --id Docker.DockerDesktop`, затем перезагрузка).
В PowerShell, из папки `Код_и_результаты`:

```powershell
docker compose build                                              # один раз, ~5 мин
docker compose run --rm lab6 python docker/reproduce.py --smoke   # проверка, ~15 мин
docker compose run --rm lab6                                      # полный повтор, ~4 ч
```

Без PyTorch (образ меньше на ~1 ГБ, пропускаются exp1c и сверка с PyTorch в `verify.py`):
`docker compose build --build-arg WITH_TORCH=0`. Если pypi.org недоступен:
`docker compose build --build-arg PIP_INDEX_URL=<адрес зеркала>`.

## Вариант 2. conda

```bat
cd /d "<путь>\Код_и_результаты"
conda env create -f docker/environment-lock.yml
conda activate lab6-repro
python docker/reproduce.py --smoke
python docker/reproduce.py
```

PyTorch для exp1c ставится отдельно: `pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu`.

## Как читать сверку

`repro_run/СВЕРКА.md` — по строке на каждую таблицу `results/`. Время счёта (столбцы `seconds`)
не сравнивается. «совпадает» — все значения равны; «почти совпадает» — отличия не больше 1e-6
(порядок суммирования в BLAS на другом процессоре); «расходится» — приводятся средние метрик
по методам до и после.
