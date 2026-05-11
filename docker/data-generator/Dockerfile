FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY data_generator/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY data_generator/ ./data_generator/
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
CMD ["python", "-m", "data_generator.kafka_producer"]
