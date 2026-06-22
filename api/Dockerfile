# filename: api/Dockerfile
# purpose:  FastAPI serving image (Phase 2A). Loads the pre-trained regression
#           and clustering Pipelines from artifacts/ at startup -- no
#           model.fit(), no notebooks/data in the image.
# version:  1.0
#
# Build context = project root (see docker-compose.yml: context: ., dockerfile: api/Dockerfile)

FROM python:3.11.9-slim

WORKDIR /app

COPY requirements-fastapi.txt .
RUN pip install --no-cache-dir -r requirements-fastapi.txt

COPY config.py .
COPY api/ api/
COPY src/ src/
COPY artifacts/regression/best_model.pkl artifacts/regression/best_model.pkl
COPY artifacts/clustering/kmeans_model.pkl artifacts/clustering/cluster_profiles.json artifacts/clustering/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
