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
RUN conda config --set remote_read_timeout_secs 600 && \
    for i in $(seq 1 3); do \
        echo "Attempt $i of 3"; \
        conda env create -f environment.yml && \
        conda clean -afy && \
        break || \
        if [ $i -lt 3 ]; then \
            echo "Retrying..."; \
            sleep 15; \
        fi; \
    done

# Now that the environment is created, continue with OS-level commands
# (Do not switch SHELL; keep default /bin/bash for build commands)

# Create non-root user for better security
RUN useradd -m appuser

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

# Expose the port for the service
EXPOSE 8501

# At runtime, ensure the conda environment is used
CMD ["conda", "run", "--no-capture-output", "-n", "inventory-env", "./start.sh"] 