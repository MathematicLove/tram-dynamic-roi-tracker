FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libx11-6 \
    libxkbcommon0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY scripts/ scripts/
COPY algorithm/ algorithm/

RUN mkdir -p models results

ENTRYPOINT ["scripts/entrypoint.sh"]
CMD ["python", "algorithm/detection.py", "--speed", "40"]
