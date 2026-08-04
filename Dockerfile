# Runtime image for the sperm_sorting research prototype.
#
# CPU by default so it builds anywhere. For the device, base this on an NVIDIA
# CUDA runtime image instead and install the matching torch build -- the
# detectors are far outside the real-time budget on CPU (see
# docs/engineering_report.md), so a GPU is not optional for live operation.
#
# pypylon is NOT installed here: the Basler SDK is licensed separately and a
# container that cannot see a USB device gains nothing from it. Add it in a
# derived image for live capture, together with `--device` passthrough.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libGL and libglib are needed by OpenCV even in its headless build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer first, so source edits do not invalidate the install.
COPY pyproject.toml README.md ./
COPY src/sperm_sorting/__init__.py src/sperm_sorting/__init__.py
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -e ".[onnx,web]"

COPY src/ src/
COPY configs/ configs/
COPY datasets/ datasets/
COPY training/ training/
COPY web/ web/
COPY scripts/ scripts/
COPY tests/ tests/

RUN pip install -e ".[onnx,web]"

# Run as a non-root user. The container has no business writing outside /app.
RUN useradd --create-home --uid 1000 sperm \
    && mkdir -p /app/runs /app/models /app/data \
    && chown -R sperm:sperm /app
USER sperm

# Fails while the instrument is uncalibrated, which is the honest answer: the
# container is healthy only when it could actually make a decision.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "from sperm_sorting.config import load_config; load_config('configs/default.yaml')" || exit 1

ENTRYPOINT ["python", "-m", "sperm_sorting.cli"]
CMD ["run", "-c", "configs/synthetic.yaml", "-n", "200"]
