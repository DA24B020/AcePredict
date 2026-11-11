# One image for the backend, MLflow server, model server and pipeline runner:
# same pinned packages everywhere = environment parity (and one copy on disk).
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp \
    ACEPREDICT_ROOT=/app \
    PYTHONPATH=/app/src:/app/backend

# Java 17 for Spark, git for DVC, curl for health checks
RUN apt-get update \
 && apt-get install -y --no-install-recommends openjdk-17-jre-headless git curl procps \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt constraints.txt ./
RUN pip install -r requirements.txt -c constraints.txt \
 && pip install pytest==8.3.4 pytest-cov==6.0.0 httpx==0.28.1 ruff==0.8.4 -c constraints.txt
RUN pip install --no-deps pathspec==0.12.1

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 755 /usr/local/bin/entrypoint.sh && chmod 666 /etc/passwd

COPY . /app
ENV HADOOP_USER_NAME=acepredict SPARK_USER=acepredict
EXPOSE 8000 5000 5001 8265 4040
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
