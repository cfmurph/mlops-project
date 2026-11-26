# Forecast API. Forecasts are read from a mounted file written by the Airflow pipeline.
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.162.0 uvicorn==0.54.0
COPY src/ src/
EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
