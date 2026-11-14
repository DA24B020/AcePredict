# The UI image: Streamlit + requests only. It never imports the ML code; it reaches the
# system only through the backend REST API (BACKEND_URL).
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HOME=/tmp

WORKDIR /frontend
COPY requirements-frontend.txt constraints.txt ./
RUN pip install -r requirements-frontend.txt -c constraints.txt
COPY frontend/ /frontend/

EXPOSE 8501 8502
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "Predict_a_Match.py", "--server.port=8501", "--server.address=0.0.0.0"]
