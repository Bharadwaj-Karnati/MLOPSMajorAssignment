# Use official Python 3.12 slim image for a lean, secure runtime
FROM python:3.12-slim

# Set working directory in container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and model artifacts (omit tests/ for lean image)
COPY src/ src/
COPY model.joblib .
COPY quant_params.joblib .
COPY unquant_params.joblib .


# Use a non-root user for security (optional, can omit if not needed)
# RUN useradd -m appuser
# USER appuser

# Entrypoint to run prediction when container starts
CMD ["python", "src/predict.py"]
