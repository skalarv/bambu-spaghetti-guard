# CPU-inference deployment image for the spaghetti guard.
# GPU hosts should run on bare metal / nvidia-container-toolkit with the
# cu128 torch build instead (see docs/INSTALL.md).
FROM python:3.11-slim

# OpenCV runtime libs (opencv-python wheels need libGL/glib on slim images)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer-cache friendly: metadata first, source after.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e .[live]

COPY config.yaml ./
# Model weights are not in the git tree — mount or COPY them at deploy time:
#   docker run -v ./models:/app/models -v ./secrets.local.txt:/app/secrets.local.txt ...
RUN mkdir -p models failure_snapshots

# Non-root, matching the systemd posture.
RUN useradd --system --no-create-home spaghetti \
    && chown -R spaghetti /app/failure_snapshots
USER spaghetti

# The guard exits 3 on camera-reconnect exhaustion / 4 on MQTT failure;
# pair with `docker run --restart on-failure`.
ENTRYPOINT ["spaghetti-guard"]
CMD ["run"]

# Liveness: enable log.heartbeat_file in config.yaml, then e.g.
#   HEALTHCHECK CMD find /app/guard.heartbeat -newermt '-15 seconds' | grep -q .
