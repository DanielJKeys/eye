# Dockerfile for EYE Platform Streamlit app
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . ./

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "eye/ui/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
