# filename: streamlit_app/Dockerfile
# purpose:  Streamlit serving image (Phase 2A). Calls the FastAPI service over
#           HTTP for all predictions (FASTAPI_URL) -- no scikit-learn/xgboost,
#           no joblib. Visual Insights page reads static artifacts/figures
#           directly (no inference/monitoring value in routing those via the API).
# version:  1.0
#
# Build context = project root (see docker-compose.yml: context: ., dockerfile: streamlit_app/Dockerfile)

FROM python:3.11.9-slim

WORKDIR /app

COPY requirements-streamlit.txt .
RUN pip install --no-cache-dir -r requirements-streamlit.txt

COPY config.py .
COPY streamlit_app/ streamlit_app/
COPY artifacts/clustering/cluster_profiles.json artifacts/clustering/pca_2d.csv artifacts/clustering/
COPY artifacts/regression/model_comparison.csv artifacts/regression/model_comparison.csv
COPY docs/figures/ docs/figures/

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app/Home.py", "--server.address=0.0.0.0", "--server.port=8501"]
