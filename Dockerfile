# Forecast API. Reads forecasts written by the Airflow pipelines from DATA_URI.
FROM python:3.12-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY config.yaml .
COPY housing/__init__.py housing/config.py housing/storage.py housing/
COPY src/ src/
EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
