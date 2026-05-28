FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY role3_mlops_devops/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY shared/ ./shared/
COPY role3_mlops_devops/api/ ./role3_mlops_devops/api/
COPY models/ ./models/

# Expose API port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "role3_mlops_devops.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
