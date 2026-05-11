FROM apache/airflow:2.10.4-python3.11
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 build-essential \
    && rm -rf /var/lib/apt/lists/*
USER airflow
WORKDIR /opt/airflow
COPY docker/ml-worker/requirements-ml.txt /tmp/requirements-ml.txt
RUN pip install --no-cache-dir -r /tmp/requirements-ml.txt
COPY data_generator/ ./data_generator/
COPY dags/ ./dags/
COPY ml/ ./ml/
ENV PYTHONPATH=/opt/airflow PYTHONUNBUFFERED=1
