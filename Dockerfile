FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-rus \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 -r requirements.txt

COPY app/ ./app/
COPY configs/ ./configs/

RUN mkdir -p uploads results

EXPOSE 8000

ENV KMP_DUPLICATE_LIB_OK=TRUE

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
