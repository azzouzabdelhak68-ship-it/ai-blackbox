FROM pytorch/pytorch:2.4.0-cuda12.1-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BLACKBOX_DIR=/blackbox \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY pyproject.toml ./
COPY blackbox/ ./blackbox/

RUN pip install --no-cache-dir -e ".[dev,search]" || pip install --no-cache-dir -e .

VOLUME ["/blackbox", "/weights"]

ENTRYPOINT ["blackbox"]
CMD ["--help"]
