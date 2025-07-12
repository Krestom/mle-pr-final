FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Устанавливаем зависимости + Prometheus-клиент и Instrumentator
RUN pip install --no-cache-dir -r requirements.txt

# Копируем приложение и данные
COPY app.py /app/app.py
RUN mkdir -p /app/data /app/models
COPY data/recommendations.parquet /app/data/
COPY data/item_popularity.parquet /app/data/
COPY data/similar_items.parquet /app/data/
COPY models/als_model.pkl /app/models/
COPY models/user_encoder.pkl /app/models/
COPY models/item_encoder.pkl /app/models/
COPY models/user_item_train.npz /app/models/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]