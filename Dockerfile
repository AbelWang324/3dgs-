ARG NERFSTUDIO_IMAGE=ghcr.io/nerfstudio-project/nerfstudio@sha256:a1525050147a5bc88c5cca15412f0f799001935f08c1b58f907b61cdd616a823
FROM ${NERFSTUDIO_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    QT_QPA_PLATFORM=offscreen \
    XDG_CACHE_HOME=/tmp/recon-cache \
    TORCH_HOME=/opt/torch-cache
RUN python -c "from torchvision.models import AlexNet_Weights; AlexNet_Weights.DEFAULT.get_state_dict(progress=False, check_hash=True)"
WORKDIR /opt/recon-worker
COPY recon_worker.py ./
ENTRYPOINT ["python", "-m", "recon_worker"]
