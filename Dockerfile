FROM python:3.15-slim-trixie AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
RUN python -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install --no-cache-dir --upgrade 'pip>=26.2' 'setuptools>=84.2.1' && \
    python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > /tmp/deps.txt && \
    /opt/venv/bin/python -m pip install --no-cache-dir -r /tmp/deps.txt
COPY qwixx_ai/ ./qwixx_ai/
RUN /opt/venv/bin/python -m pip install --no-cache-dir --no-deps . && /opt/venv/bin/python -m pip check

FROM python:3.15-slim-trixie AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PATH=/opt/venv/bin:$PATH
ARG QWIXX_GIT_VERSION=dev
ARG QWIXX_GIT_SHA=unknown
ARG QWIXX_BUILD_DATE=unknown
ENV QWIXX_GIT_VERSION=${QWIXX_GIT_VERSION} \
    QWIXX_GIT_SHA=${QWIXX_GIT_SHA} \
    QWIXX_BUILD_DATE=${QWIXX_BUILD_DATE}
WORKDIR /app
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
COPY qwixx_ai/ ./qwixx_ai/
# Validate the runtime dependencies and native modules before switching user.
RUN python -m pip check && python -c "import sqlite3, ssl, torch, numpy; assert torch.tensor([1,2]).sum().item()==3; assert sqlite3.connect(':memory:').execute('select 1').fetchone()==(1,); ssl.create_default_context()" && \
    mkdir -p /app/runs && chown -R 10001:10001 /app
USER 10001:10001
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=8s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=5)" || exit 1
CMD ["python","-m","qwixx_ai.distributed","--host","0.0.0.0","--port","8765"]
