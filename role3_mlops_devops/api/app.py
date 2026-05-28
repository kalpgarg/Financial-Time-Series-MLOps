"""
FastAPI application serving model predictions.

Endpoints:
  POST /predict  → accepts PredictionRequest, returns PredictionResponse
  GET  /health   → liveness check
  GET  /metrics  → Prometheus metrics endpoint
"""

# TODO: Load model from MODEL_PATH, define /predict endpoint
# Use shared.schemas for PredictionRequest, PredictionResponse
# Use shared.config for API_HOST, API_PORT, MODEL_PATH
