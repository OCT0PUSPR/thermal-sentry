# thermal-sentry container image.
#
# This image is arm64-friendly: the python:3.11-slim-bookworm base has official
# linux/arm64 builds, so it runs natively on a Raspberry Pi 4/5 (64-bit OS) as
# well as on x86_64 laptops/CI. Build for the Pi explicitly with:
#
#   docker buildx build --platform linux/arm64 -t thermal-sentry:arm64 .
#
# By default this image runs the SIMULATE pipeline + dashboard, so it works with
# zero hardware. To drive the real MLX90640 you must (a) run on a Pi, (b) install
# the Pi extras, and (c) pass the I2C device into the container -- see README.

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TS_SOURCE=simulate \
    TS_WEB_HOST=0.0.0.0 \
    TS_WEB_PORT=8000

# Minimal system deps. (No heavy CV stack; pure-numpy fallbacks cover us.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the project and install it.
COPY pyproject.toml README.md LICENSE ./
COPY thermalsentry ./thermalsentry
RUN pip install --no-deps .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Default: serve the dashboard against the simulator (works with no hardware).
CMD ["thermal-sentry", "run", "--source", "simulate", "--web"]
