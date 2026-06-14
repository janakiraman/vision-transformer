"""
Runs at DOCKER BUILD time (huggingface mode only) to bake the model
into the image at /app/model. Skipped if you COPY a local model instead.
"""
import os

from transformers import ViTForImageClassification, ViTImageProcessor

MODEL_NAME = os.getenv("HF_MODEL_NAME", "google/vit-base-patch16-224")
OUT = "/app/model"

print(f"Downloading {MODEL_NAME} -> {OUT}")
ViTForImageClassification.from_pretrained(MODEL_NAME).save_pretrained(OUT)
ViTImageProcessor.from_pretrained(MODEL_NAME).save_pretrained(OUT)
print("Done.")
