FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
WORKDIR /app
COPY pyproject.toml README.md ./
COPY qwixx_ai/ ./qwixx_ai/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .
RUN mkdir -p /app/runs && chown -R 10001:10001 /app
USER 10001:10001
EXPOSE 8765
CMD ["python","-m","qwixx_ai.web","--host","0.0.0.0","--port","8765"]
