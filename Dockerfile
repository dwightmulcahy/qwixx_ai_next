FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
ARG QWIXX_GIT_VERSION=dev
ARG QWIXX_GIT_SHA=unknown
ARG QWIXX_BUILD_DATE=unknown
ENV QWIXX_GIT_VERSION=${QWIXX_GIT_VERSION} \
    QWIXX_GIT_SHA=${QWIXX_GIT_SHA} \
    QWIXX_BUILD_DATE=${QWIXX_BUILD_DATE}
WORKDIR /app
COPY pyproject.toml README.md ./
# Dependencies (torch, numpy) first, from pyproject only, so a source change reuses this layer
RUN pip install --no-cache-dir --upgrade pip && \
    python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > /tmp/deps.txt && \
    pip install --no-cache-dir -r /tmp/deps.txt && rm /tmp/deps.txt
COPY qwixx_ai/ ./qwixx_ai/
RUN pip install --no-cache-dir --no-deps .
RUN mkdir -p /app/runs && chown -R 10001:10001 /app
USER 10001:10001
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=8s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=5)" || exit 1
CMD ["python","-m","qwixx_ai.distributed","--host","0.0.0.0","--port","8765"]
