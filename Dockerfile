# Stage 1: Builder
# This stage will now be used for training as well
FROM python:3.12-slim AS builder

WORKDIR /app

# Install all dependencies needed for training
USER root
RUN apt-get update && apt-get install -y --no-install-recommends build-essential
COPY requirements.txt .
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -r requirements-dev.txt

# Copy all source, data, and config needed for training
COPY src/ ./src
COPY data/ ./data
COPY config.yaml .

# Set path and run training to generate the artifact
ENV PYTHONPATH=/app
RUN python -m src.credit_risk.train

# Stage 2: Final Image
FROM python:3.12-slim AS final

# Create a non-root user and group (Debian syntax)
RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup --home /home/appuser --disabled-password appuser

USER appuser

WORKDIR /home/appuser/app

# Copy the application source code, preserving the package structure.
COPY --chown=appuser:appgroup src/ ./src/
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/

# Copy other necessary files into their expected locations.
COPY docs/openapi.json ./credit_risk/

# Copy the generated model artifact from the builder stage
COPY --from=builder /app/artifacts/model.joblib ./artifacts/

# Expose the port the app runs on
EXPOSE 9696

# Make Python aware of the installed packages
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/home/appuser/app

# Healthcheck using Python stdlib (no curl needed)
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9696/ping')" || exit 1

# Run with Gunicorn
CMD ["gunicorn", "--bind=0.0.0.0:9696", "src.credit_risk.predict:app"]