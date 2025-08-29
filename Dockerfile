# Stage 1: Builder
# This stage installs dependencies and is discarded later.
FROM python:3.12-slim as builder

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies into a temporary location
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final Image
# This is the lean, final image that will run in production.
FROM python:3.12-slim

# Create a non-root user
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
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

# Healthcheck to ensure the application is running
HEALTHCHECK CMD curl --fail http://localhost:9696/ || exit 1

# Define the command to run the application using Gunicorn for production
CMD ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]