# Pokemon RL Bot - CPU Version
# Multi-stage build for production deployment

# Build stage
FROM python:3.9-slim as builder

# Set build arguments
ARG VERSION=latest
ARG BUILD_DATE
ARG VCS_REF

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    libx11-dev \
    libxext-dev \
    libxrandr-dev \
    libxcursor-dev \
    libxi-dev \
    libxinerama-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libasound2-dev \
    libpulse-dev \
    libopenal-dev \
    libsdl2-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install VBA-M emulator
RUN apt-get update && apt-get install -y vbam && rm -rf /var/lib/apt/lists/*

# Production stage
FROM python:3.9-slim as production

# Set build arguments and labels
ARG VERSION=latest
ARG BUILD_DATE
ARG VCS_REF

LABEL org.opencontainers.image.title="Pokemon RL Bot" \
      org.opencontainers.image.description="Reinforcement Learning bot for Pokemon Leaf Green" \
      org.opencontainers.image.version="$VERSION" \
      org.opencontainers.image.created="$BUILD_DATE" \
      org.opencontainers.image.revision="$VCS_REF" \
      org.opencontainers.image.licenses="MIT"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    vbam \
    libx11-6 \
    libxext6 \
    libxrandr2 \
    libxcursor1 \
    libxi6 \
    libxinerama1 \
    libgl1-mesa-glx \
    libglu1-mesa \
    libasound2 \
    libpulse0 \
    libopenal1 \
    libsdl2-2.0-0 \
    libgtk-3-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r pokemon && useradd -r -g pokemon -s /bin/bash pokemon

# Create app directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY requirements.txt config.json ./

# Create necessary directories
RUN mkdir -p logs models roms saves data/screenshots data/recordings && \
    chown -R pokemon:pokemon /app

# Set environment variables
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=src.web.app \
    ENV=production

# Switch to non-root user
USER pokemon

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:5000/api/status || exit 1

# Expose port
EXPOSE 5000

# Default command
CMD ["python", "scripts/monitor.py", "--config", "config.json", "--host", "0.0.0.0", "--port", "5000"]
