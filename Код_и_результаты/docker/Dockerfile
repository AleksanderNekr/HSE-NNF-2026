# Окружение для повтора расчётов (NNF / NNFA). Код и данные в образ не входят:
# папка «Код_и_результаты» монтируется в /work (см. docker-compose.yml).
FROM python:3.11-slim-bookworm

# Liberation Serif — шрифт рисунков (как Times New Roman)
RUN apt-get update \
 && apt-get install -y --no-install-recommends fonts-liberation \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/mpl \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Зеркало PyPI, если pypi.org недоступен:  docker compose build --build-arg PIP_INDEX_URL=<адрес>
ARG PIP_INDEX_URL=https://pypi.org/simple
# PyTorch (CPU) нужен только для сверки с кодом авторов (exp1c и проверка 3 в verify.py).
# Без него:  docker compose build --build-arg WITH_TORCH=0
ARG WITH_TORCH=1
ARG TORCH_VERSION=2.15.0

COPY requirements-lock.txt /tmp/requirements-lock.txt
RUN pip install --retries 10 --index-url "$PIP_INDEX_URL" -r /tmp/requirements-lock.txt
RUN if [ "$WITH_TORCH" = "1" ]; then \
      pip install --retries 10 --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"; \
    fi

WORKDIR /work
CMD ["python", "docker/reproduce.py"]
