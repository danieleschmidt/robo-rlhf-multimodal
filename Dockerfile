FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml ./
COPY robo_rlhf/ ./robo_rlhf/

RUN pip install --no-cache-dir -e .

EXPOSE 8080

CMD ["python", "-m", "robo_rlhf.preference_server"]