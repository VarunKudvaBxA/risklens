FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e .
# Train inside the image so the container is self-contained (demo data). Mount a trained model instead in production.
RUN python -m risklens.train --no-mlflow --n-trials 5
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
