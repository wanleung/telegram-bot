FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
COPY config.example.yaml ./

RUN mkdir -p data

ENV CONFIG_PATH=/app/config.yaml

CMD ["python", "bot.py"]
