FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git git-lfs gdal-bin libgdal-dev

# Clone and install FLAIR-HUB
RUN git clone https://github.com/IGNF/FLAIR-HUB.git && \
    cd FLAIR-HUB && \
    pip install -e .

# Download model weights
RUN git lfs install && \
    git clone https://huggingface.co/IGNF/FLAIR-HUB_LC-A_RGBI_swinbase-upernet

COPY config_nrw_inference.yaml /app/
COPY extract_green_areas.py /app/

# Default command
CMD ["flairhub_zonal", "--config", "config_nrw_inference.yaml"]
