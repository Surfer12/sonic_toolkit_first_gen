# Reverse Koopman Penetration Testing Framework - Java Implementation
# Dockerfile for development and deployment

FROM openjdk:21-slim

# Set environment variables
ENV JAVA_HOME=/usr/local/openjdk-21
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    maven \
    gradle \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy source code
COPY . .

# Create directories for outputs and reports
RUN mkdir -p /app/out \
    /app/reports \
    /app/logs \
    /app/data

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Set default command
CMD ["bash"]

# Labels for metadata
LABEL maintainer="Jumping Quail Solutions"
LABEL version="1.0"
LABEL description="Reverse Koopman Penetration Testing Framework - Java Implementation"
LABEL repository="https://github.com/jumpingquail/reverse-koopman-pentest"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD java -version || exit 1
