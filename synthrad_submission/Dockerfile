# =============================================================================
# SynthRAD2025 Grand Challenge - MR-to-CT Synthesis Docker Container
#
# Build context: repository root (fullbody-sCT/)
# Build command: docker build -t synthrad_algorithm -f synthrad_submission/Dockerfile .
# =============================================================================

FROM condaforge/miniforge3:latest

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (Grand Challenge convention)
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

# Standard directories
RUN mkdir -p /opt/algorithm /input /output /tmp/pipeline \
    && chown -R algorithm:algorithm /opt/algorithm /input /output /tmp/pipeline

WORKDIR /opt/algorithm

# ---------------------------------------------------------------------------
# Conda environments (expensive layers -- cached early)
# ---------------------------------------------------------------------------

COPY preprocessing/preprocessing_env_docker.yml /opt/algorithm/envs/
COPY training/inference_env_docker.yaml          /opt/algorithm/envs/

# Preprocessing env: Python 3.7, CPU PyTorch 1.10.0
RUN mamba env create -f /opt/algorithm/envs/preprocessing_env_docker.yml \
    && mamba clean -afy

# Inference env: Python 3.6, CUDA PyTorch 1.10.1
# Override name: YAML says "pytorch-CycleGAN-and-pix2pix_new" but the
# pipeline script expects "inference_env_docker"
RUN mamba env create -f /opt/algorithm/envs/inference_env_docker.yaml \
        -n inference_env_docker \
    && mamba clean -afy

# ---------------------------------------------------------------------------
# Pipeline code (directory structure must match repo for import resolution)
# ---------------------------------------------------------------------------

COPY --chown=algorithm:algorithm \
    preprocessing/preprocessing_synthrad/ \
    /opt/algorithm/preprocessing/preprocessing_synthrad/

# training/data/ is imported by inference_synth.py
COPY --chown=algorithm:algorithm \
    training/data/ \
    /opt/algorithm/training/data/

COPY --chown=algorithm:algorithm \
    training_cut/ \
    /opt/algorithm/training_cut/

# ---------------------------------------------------------------------------
# Model checkpoints
# ---------------------------------------------------------------------------

COPY --chown=algorithm:algorithm \
    data/checkpoints/cut_synthrad_allregions_final/ \
    /opt/algorithm/checkpoints/cut_synthrad_allregions_final/

# ---------------------------------------------------------------------------
# Pipeline script and Docker entrypoint
# ---------------------------------------------------------------------------

COPY --chown=algorithm:algorithm synthrad2025_inference.sh /opt/algorithm/
COPY --chown=algorithm:algorithm synthrad_submission/docker_entrypoint.sh /opt/algorithm/

# Fix Windows CRLF line endings and set executable
RUN sed -i 's/\r$//' /opt/algorithm/synthrad2025_inference.sh \
                      /opt/algorithm/docker_entrypoint.sh \
    &&  chmod +x /opt/algorithm/synthrad2025_inference.sh \
             /opt/algorithm/docker_entrypoint.sh

# # Ensure conda envs are readable by non-root user
# RUN chmod -R a+rX /opt/conda/envs/

ARG IMAGE_VERSION
LABEL org.opencontainers.image.revision=$IMAGE_VERSION
RUN echo $IMAGE_VERSION > /image_version.txt

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

USER algorithm

ENV TASK_TYPE=mri
ENV EXECUTE_IN_DOCKER=1

ENTRYPOINT ["/opt/algorithm/docker_entrypoint.sh"]
