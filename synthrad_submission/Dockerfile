# =============================================================================
# SynthRAD2025 Grand Challenge - MR-to-CT Synthesis Docker Container
#
# Build context: repository root (fullbody-sCT/)
# Build command:
#   docker build -t synthrad_algorithm -f synthrad_submission/Dockerfile \
#       --build-arg MODEL_TYPE=CUT --build-arg BODYREGION_TYPE=allregions .
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Select checkpoints based on MODEL_TYPE
# ---------------------------------------------------------------------------
FROM alpine:latest AS checkpoint-selector

ARG MODEL_TYPE=CUT

# Copy ALL checkpoints to builder stage
COPY data/checkpoints/ /all_checkpoints/

# Select only the 4 checkpoints matching MODEL_TYPE
# Patterns: CUT=cut_synthrad_*, cycleGAN=cyclegan_*, pix2pix=pix2pix_synthrad_*
RUN mkdir -p /selected_checkpoints && \
    case "${MODEL_TYPE}" in \
        CUT)      cp -r /all_checkpoints/cut_synthrad_* /selected_checkpoints/ ;; \
        cycleGAN) cp -r /all_checkpoints/cyclegan_* /selected_checkpoints/ ;; \
        pix2pix)  cp -r /all_checkpoints/pix2pix_synthrad_* /selected_checkpoints/ ;; \
        *)        echo "ERROR: Unknown MODEL_TYPE: ${MODEL_TYPE}" && exit 1 ;; \
    esac && \
    echo "Selected checkpoints for MODEL_TYPE=${MODEL_TYPE}:" && \
    ls -la /selected_checkpoints/

# ---------------------------------------------------------------------------
# Stage 2: Final image
# ---------------------------------------------------------------------------
FROM condaforge/miniforge3:latest

# Re-declare ARGs for use in final stage (ARGs don't persist across stages)
ARG MODEL_TYPE=CUT
ARG BODYREGION_TYPE=allregions

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

COPY --chown=algorithm:algorithm \
    training/ \
    /opt/algorithm/training/

COPY --chown=algorithm:algorithm \
    training_cut/ \
    /opt/algorithm/training_cut/

# ---------------------------------------------------------------------------
# Model checkpoints - ONLY the 4 matching MODEL_TYPE (from builder stage)
# ---------------------------------------------------------------------------

COPY --from=checkpoint-selector --chown=algorithm:algorithm \
    /selected_checkpoints/ \
    /opt/algorithm/checkpoints/

# ---------------------------------------------------------------------------
# Pipeline script and Docker entrypoint
# ---------------------------------------------------------------------------

COPY --chown=algorithm:algorithm synthrad_submission/synthrad2025_inference.sh /opt/algorithm/
COPY --chown=algorithm:algorithm synthrad_submission/docker_entrypoint.sh /opt/algorithm/
COPY --chown=algorithm:algorithm synthrad_submission/load_input.py /opt/algorithm/synthrad_submission/
COPY --chown=algorithm:algorithm synthrad_submission/find_model_ex1.py /opt/algorithm/synthrad_submission/
COPY --chown=algorithm:algorithm synthrad_submission/provide_output.py /opt/algorithm/synthrad_submission/

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
# Make MODEL_TYPE and BODYREGION_TYPE available at runtime
ENV MODEL_TYPE=${MODEL_TYPE}
ENV BODYREGION_TYPE=${BODYREGION_TYPE}

ENTRYPOINT ["/opt/algorithm/docker_entrypoint.sh"]
