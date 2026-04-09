FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p artifacts/raw \
    artifacts/processed \
    artifacts/cleaned \
    artifacts/features \
    artifacts/predictions \
    artifacts/metrics \
    data \
    docs

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
