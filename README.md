# ViT on AKS — model baked into the image

```
app/main.py          FastAPI inference server (huggingface or pytorch mode)
download_model.py    Downloads HF model at docker build time
model/               Put your model files here (baked in at build)
Dockerfile           Bakes model into image
k8s/                 deployment.yaml, service.yaml, hpa.yaml
```

## 1. Prepare the model

**HuggingFace ViT** — either:
- Do nothing: the build auto-downloads `google/vit-base-patch16-224` (override with `--build-arg HF_MODEL_NAME=...`), or
- Copy your fine-tuned checkpoint into `model/` (`config.json`, `model.safetensors`, `preprocessor_config.json`):
  ```python
  model.save_pretrained("model/"); processor.save_pretrained("model/")
  ```

**Plain PyTorch (torchvision vit_b_16)** — put files in `model/`:
- `model.pth` — state_dict (`torch.save(model.state_dict(), "model.pth")`)
- `labels.json` — optional, `{"0": "cat", "1": "dog"}`

> Different architecture than `vit_b_16`? Edit `load_pytorch()` in `app/main.py`.

## 2. Build and test locally (optional)

```bash
# HuggingFace
docker build -t vit-api:v1 .

# PyTorch
docker build --build-arg MODEL_TYPE=pytorch -t vit-api:v1 .

docker run -p 8000:8000 vit-api:v1
curl -F "file=@cat.jpg" http://localhost:8000/predict
```

## 3. Push to Azure Container Registry

```bash
az login
RG=vit-rg; ACR=vitacr$RANDOM; LOC=eastus            # pick your names
az group create -n $RG -l $LOC
az acr create -n $ACR -g $RG --sku Basic

# Build IN Azure (no local docker needed):
az acr build -r $ACR -t vit-api:v1 .
# pytorch mode:
# az acr build -r $ACR -t vit-api:v1 --build-arg MODEL_TYPE=pytorch .

# (or build locally and push)
# az acr login -n $ACR
# docker tag vit-api:v1 $ACR.azurecr.io/vit-api:v1
# docker push $ACR.azurecr.io/vit-api:v1
```

## 4. Create AKS cluster and attach ACR

```bash
AKS=vit-aks
az aks create -n $AKS -g $RG --node-count 2 -s Standard_D4s_v3 \
  --attach-acr $ACR --generate-ssh-keys
# existing cluster: az aks update -n $AKS -g $RG --attach-acr $ACR

az aks get-credentials -n $AKS -g $RG
```

## 5. Deploy

```bash
# put your ACR name into the manifest
sed -i '' "s/<ACR_NAME>/$ACR/" k8s/deployment.yaml     # macOS (Linux: sed -i)
# pytorch mode: set MODEL_TYPE env to "pytorch" in deployment.yaml

kubectl apply -f k8s/
kubectl rollout status deployment/vit-api
kubectl get pods -l app=vit-api
```

## 6. Test

```bash
EXTERNAL_IP=$(kubectl get svc vit-api -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl http://$EXTERNAL_IP/healthz
curl -F "file=@cat.jpg" http://$EXTERNAL_IP/predict
```

Response:
```json
{"predictions": [{"label": "tabby, tabby cat", "class_id": 281, "score": 0.91}, ...]}
```

## GPU variant

1. GPU node pool: `az aks nodepool add -g $RG --cluster-name $AKS -n gpupool -c 1 -s Standard_NC6s_v3 --node-taints sku=gpu:NoSchedule`
2. Dockerfile: base image `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`; remove the CPU-only torch install line.
3. deployment.yaml: add `nvidia.com/gpu: 1` under `resources.limits` and a toleration for `sku=gpu`.

## Updating the model

Model is baked in, so a new model = new image:
```bash
az acr build -r $ACR -t vit-api:v2 .
kubectl set image deployment/vit-api vit-api=$ACR.azurecr.io/vit-api:v2
```
