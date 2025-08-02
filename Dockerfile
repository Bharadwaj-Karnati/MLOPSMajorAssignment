FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY tests/ tests/
COPY model.joblib .
COPY quant_params.joblib .
COPY unquant_params.joblib .

CMD ["python", "src/predict.py"]

