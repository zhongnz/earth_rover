FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
  && python -m playwright install --with-deps webkit

COPY . .

EXPOSE 8000

CMD ["hypercorn", "main:app", "--bind", "0.0.0.0:8000"]
