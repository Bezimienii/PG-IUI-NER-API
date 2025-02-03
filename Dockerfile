FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN set -x && \
    apt-get update && \
    apt-get install --no-install-recommends --assume-yes \
      build-essential \
    && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create user
ARG UID=1000
ARG GID=1000
RUN set -x && \
    groupadd -g "${GID}" python && \
    useradd --create-home --no-log-init -u "${UID}" -g "${GID}" python &&\
    chown python:python -R /app


USER python

RUN mkdir -p /app/src/sync && chmod -R 777 /app/src/sync
RUN touch /app/src/sync/counter.lock && chmod 777 /app/src/sync/counter.lock
RUN chown -R python:python /app/src/sync
# Install python dependencies
COPY requirements.txt . 
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application code
COPY src src

# Create directories for local storage
RUN mkdir -p /app/databases /app/models /app/tmp && \
    chown -R python:python /app/databases /app/models /app/tmp

# Set environment variables for local directories
ENV DATABASES_DIR="/app/databases"
ENV MODELS_DIR="/app/models"
ENV TMP_DIR="/app/tmp"

# don't buffer Python output
ENV PYTHONUNBUFFERED=1

# Add pip's user base to PATH
ENV PATH="$PATH:/home/python/.local/bin"

# Expose port (this will still be dynamic)
EXPOSE 8080

# Change the CMD to use sh -c for background execution and logging
CMD ["sh", "-c", "nohup python -m fastapi run src/main.py --port 8080 > logi.txt 2>&1 &"]
