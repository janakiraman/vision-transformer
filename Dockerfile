# syntax=docker/dockerfile:1
# =============================================================
# Bakes the ViT model INTO the image at build time.
# Two modes (build arg MODEL_TYPE):
#   huggingface (default): downloads HF model during build
#                          (or COPY your fine-tuned HF checkpoint)
#   pytorch              : copies local model/model.pth (+ labels.json)
#
# Build examples:
#   docker build -t vit-api .                                  # HF, downloads at build
#   docker build --build-arg HF_MODEL_NAME=google/vit-base-patch16-224 -t vit-api .
#   docker build --build-arg MODEL_TYPE=pytorch -t vit-api .   # needs ./model/model.pth
#
# GPU: change base image to pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
#      and remove the CPU-only torch index URL below.
# =============================================================

FROM python:3.11-slim AS base

ARG MODEL_TYPE=huggingface
ARG HF_MODEL_NAME=google/vit-base-patch16-224

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_TYPE=${MODEL_TYPE} \
    MODEL_DIR=/app/model

WORKDIR /app

# Install deps first (better layer caching). CPU-only torch keeps image small.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ---- bake the model into the image ----
# Local files in ./model/ (HF checkpoint dir OR model.pth) are copied in.
# If huggingface mode and no local weights were copied, download at build time.
COPY model/ /app/model/
COPY download_model.py .
RUN if [ "$MODEL_TYPE" = "huggingface" ] && [ ! -f /app/model/config.json ]; then \
        HF_MODEL_NAME=${HF_MODEL_NAME} python download_model.py ; \
    fi && \
    if [ "$MODEL_TYPE" = "pytorch" ] && [ ! -f /app/model/model.pth ]; then \
        echo "ERROR: MODEL_TYPE=pytorch requires ./model/model.pth" && exit 1 ; \
    fi

# ---- app code ----
COPY app/ /app/app/

# Non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
