# Use conda base image instead of plain python
FROM continuumio/miniconda3:latest

# Install NVIDIA container toolkit dependencies
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Install system dependencies (curl)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set an OpenMP environment variable to avoid potential runtime issues
ENV KMP_INIT_AT_FORK=FALSE

# Set working directory
WORKDIR /app

# Copy environment file
COPY environment.yml .

# Create conda environment and install dependencies
RUN conda env create -f environment.yml && \
    echo "conda activate inventory-env" >> ~/.bashrc && \
    conda init bash

# Make RUN commands use the new environment
SHELL ["/bin/bash", "--login", "-c"]

# Create non-root user for better security
RUN useradd -m appuser

# Set up conda for the non-root user
COPY --chown=appuser:appuser environment.yml /home/appuser/
RUN mkdir -p /home/appuser/.conda && \
    chown -R appuser:appuser /home/appuser/.conda && \
    conda init bash && \
    echo "conda activate inventory-env" >> /home/appuser/.bashrc

# Copy application code
COPY . .

# Set permissions for the application and mlruns folder
RUN chown -R appuser:appuser /app && \
    chmod +x start.sh && \
    mkdir -p /app/mlruns && \
    chown -R appuser:appuser /app/mlruns && \
    chmod -R 777 /app/mlruns && \
    chmod -R 777 /app

# Switch to non-root user
USER appuser
SHELL ["/bin/bash", "--login", "-c"]

# Expose the port for the service
EXPOSE 8501

# Run the application with conda environment activated
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate inventory-env && ./start.sh"] 