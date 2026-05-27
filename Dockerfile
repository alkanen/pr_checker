FROM python:3.10-slim

WORKDIR /app

RUN adduser --disabled-password --gecos "" appuser

COPY pyproject.toml ./
COPY pr_checker/ ./pr_checker/

RUN pip install --no-cache-dir . && chown appuser /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000
ENV ROOT_PATH=

EXPOSE 8000

USER appuser

CMD ["python", "-m", "pr_checker"]
