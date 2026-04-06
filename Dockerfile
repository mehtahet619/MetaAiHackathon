FROM python:3.11-slim

WORKDIR /app

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models.py .
COPY server/ ./server/

EXPOSE 7860

CMD ["python3", "-c", "import sys,os; sys.path.insert(0,'.'); sys.path.insert(0,'./server'); from server.app import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=7860)"]