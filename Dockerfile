# Stage 1: Builder
FROM python:3.13-slim AS builder

WORKDIR /app

# Copy the requirements file (generated in CI)
COPY requirements.txt .

# Install dependencies into a temporary location
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final Image
FROM python:3.13-slim AS final

# Create a non-root user and group (Debian syntax)
RUN addgroup --system appgroup \
    && adduser --system --ingroup appgroup --home /home/appuser --disabled-password appuser

USER appuser

WORKDIR /home/appuser/app

# Copy installed packages from the builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy the application code and the serialized model
COPY ["predict.py", "model.joblib", "./"]

# Expose the port the app runs on
EXPOSE 9696

# Make Python aware of the installed packages
ENV PATH=/home/appuser/.local/bin:$PATH

# Healthcheck using Python stdlib (no curl needed)
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9696/ping')" || exit 1

# Run with Gunicorn
CMD ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]