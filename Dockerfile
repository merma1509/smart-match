FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-rus \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Устанавливаем зависимости из pyproject.toml
COPY pyproject.toml .
RUN pip install --no-cache-dir --timeout 120 .

# Копируем предварительно скачанные EasyOCR модели
RUN mkdir -p /root/.EasyOCR/model
COPY app/models/easyocr/craft_mlt_25k.pth /root/.EasyOCR/model/
COPY app/models/easyocr/cyrillic_g2.pth /root/.EasyOCR/model/

COPY app/ ./app/
COPY configs/ ./configs/

RUN mkdir -p uploads results

EXPOSE 8000
ENV KMP_DUPLICATE_LIB_OK=TRUE

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
