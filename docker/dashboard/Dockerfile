FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY dashboard/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY data_generator/ ./data_generator/
COPY dashboard/app.py ./dashboard/app.py
COPY ml/ ./ml/
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", \
     "--server.address=0.0.0.0", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
