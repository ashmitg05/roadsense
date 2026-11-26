# Inference image: ONNX Runtime + OpenCV only. No PyTorch, no Ultralytics.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080 \
    MODEL_PATH=/app/models/model.onnx

WORKDIR /app
COPY requirements-serve.txt .
# --no-deps for the tracker: skips matplotlib, PyAV and a 2nd (GUI) OpenCV it never imports (~0.5 GB)
RUN pip install --no-cache-dir -r requirements-serve.txt \
 && pip install --no-cache-dir --no-deps supervision==0.32.0 trackers==2.7.0

COPY src/__init__.py src/__init__.py
COPY src/inference src/inference
COPY models/model.onnx models/model.onnx

RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8080
CMD exec uvicorn src.inference.api:app --host 0.0.0.0 --port ${PORT} --workers 1
