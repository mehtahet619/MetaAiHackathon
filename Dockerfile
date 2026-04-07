FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uvicorn fastapi openenv-core

COPY models.py .
COPY server/ ./server/
COPY traffic-control-sim.html ./static/index.html

EXPOSE 7860

CMD ["python3", "server/app.py", "--host", "0.0.0.0", "--port", "7860"]