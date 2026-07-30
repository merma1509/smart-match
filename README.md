# Smart Match

**AI-powered document intelligence for historical Russian metrical books**

Smart Match is a REST API for automatic extraction of structured genealogical data (names, dates, places) from scans of historical metrical books using hybrid OCR and LLM-assisted extraction.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Development](#development)

---

## Overview

**Input**: photos/scans of metrical book pages (JPG, PNG)  
**Output**: structured JSON with extracted records

### Supported Record Types

| Type         | Keywords                  | Extracted Fields                     |
| ------------ | ------------------------- | ------------------------------------ |
| **Birth**    | родился, крещён, родилася | person, date, place, parents         |
| **Death**    | умер, скончался, погребён | person, date, place, age             |
| **Marriage** | венчался, обручен, брак   | groom, bride, date, place, witnesses |

---

## Architecture

### Pipeline Flow

```
Image Upload → Preprocessing (OpenCV) → Layout Detection → OCR Engine
                                                              │
                                          ┌───────────────────┼───────────────────┐
                                          │                   │                   │
                                     Fine-tuned         EasyOCR              Tesseract
                                       TrOCR             (cyrillic)            (rus)
                                          │                   │                   │
                                          └───────────────────┴───────────────────┘
                                                              │
                                                              ▼
                                                    Text Cleanup & Correction
                                                              │
                                                              ▼
                                                    Record Type Detection
                                                   (keyword scoring)
                                                              │
                                         ┌─────────────────────┼─────────────────────┐
                                         │                     │                     │
                                    Birth Extractor      Death Extractor     Marriage Extractor
                                         │                     │                     │
                                         └─────────────────────┴─────────────────────┘
                                                              │
                                                              ▼
                                                    Confidence Check
                                                    (threshold: 0.7)
                                                              │
                              ┌──────────────────────────────┼──────────────────────────────┐
                              │                              │                              │
                         Rule-based                     LLM Extractor                   Hybrid
                         (high conf)                   (low conf / force)             (default)
                              │                              │                              │
                              └──────────────────────────────┴──────────────────────────────┘
                                                              │
                                                              ▼
                                                    Entity Resolution
                                                              │
                                                              ▼
                                                         JSON Output
```

### Pipeline Details

1. **Preprocessing** — color/illumination normalization, deskewing, denoising
2. **OCR Engine** — cascading: fine-tuned TrOCR → EasyOCR (cyrillic) → Tesseract
3. **Text Cleanup** — correction of typical OCR errors for old Cyrillic
4. **Record Type Detection** — identification by keyword scoring
5. **Information Extraction** — rule-based regex + LLM for complex records
6. **Entity Resolution** — deduplication and normalization

---

## Tech Stack

| Component            | Technology                    | Version/Parameters         |
| -------------------- | ----------------------------- | -------------------------- |
| **API Framework**    | FastAPI + Uvicorn             | Starlette-based            |
| **OCR Primary**      | Fine-tuned TrOCR              | 334M params                |
| **OCR Secondary**    | EasyOCR                       | cyrillic_g2.pth            |
| **OCR Fallback**     | Tesseract OCR                 | Language: rus              |
| **LLM Backend**      | Ollama / OpenAI / HuggingFace | llama3.1:8b (configurable) |
| **Preprocessing**    | OpenCV, NumPy                 | cv2, scikit-image          |
| **Config**           | Pydantic BaseModel            | Type-safe settings         |
| **Containerization** | Docker + docker-compose       | Python 3.11-slim           |

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ (for local development)
- Ollama (for LLM extraction)

### 1. Run with Docker

```bash
git clone https://github.com/mnijonshuti/smart-match.git
cd smart-match
docker compose up --build -d
curl http://localhost:8000/health
```

### 2. Setup Ollama (for LLM)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull llama3.1:8b

# Start Ollama
ollama serve
```

### 3. Test Extraction

```bash
curl -s -X POST http://localhost:8000/extract \
  -F "file=@path/to/image.jpg" | jq .
```

### Make Commands

```bash
make help      # Show all commands
make up        # Start containers
make down      # Stop containers
make restart   # Restart with rebuild
make logs      # View logs
make health    # Check API health
```

---

## API Endpoints

### GET /health

```json
{
  "status": "healthy",
  "service": "Smart Match",
  "version": "1.1.0",
  "issues": null
}
```

### POST /extract

Extract data from image (multipart/form-data, file field: image)

```json
{
  "request_id": "a1b2c3d4",
  "records": [
    {
      "record_type": "birth",
      "person": {
        "name": "Иван",
        "patronymic": "Иванович",
        "surname": "Петров"
      },
      "date": { "day": 15, "month": 3, "year": 1847 },
      "place": {
        "settlement": "село Покровское",
        "district": "Касимовский уезд"
      },
      "parents": {
        "father": { "name": "Иван", "surname": "Петров" },
        "mother": { "name": "Анна", "surname": "Петрова" }
      }
    }
  ],
  "_extraction": {
    "method": "hybrid",
    "average_confidence": 0.0,
    "llm_calls": 2,
    "llm_provider": "ollama"
  }
}
```

### GET /results/{request_id}

Retrieve previously saved result.

### POST /batch-extract

Batch processing (multipart/form-data, files field).

---

## Configuration

Main settings in `app/core/config.py`:

### OCR Settings

| Parameter                  | Default | Description                  |
| -------------------------- | ------- | ---------------------------- |
| `ocr_confidence_threshold` | 0.7     | Threshold for LLM triggering |
| `ocr_use_gpu`              | false   | Use GPU for EasyOCR          |
| `tesseract_lang`           | rus     | Tesseract language           |

### LLM Settings

| Parameter                  | Default                | Description                 |
| -------------------------- | ---------------------- | --------------------------- |
| `llm_provider`             | ollama                 | ollama, openai, huggingface |
| `llm_model_name`           | llama3.1:8b            | Model name                  |
| `llm_api_base_url`         | http://localhost:11434 | API base URL                |
| `llm_confidence_threshold` | 0.7                    | LLM activation threshold    |

### Preprocessing

| Parameter              | Default | Description              |
| ---------------------- | ------- | ------------------------ |
| `max_image_dimension`  | 3000    | Max image dimension (px) |
| `border_removal_width` | 10      | Border removal width     |

### Environment Variables

```bash
SMART_MATCH_ENV=production
SMART_MATCH_LLM_PROVIDER=openai
SMART_MATCH_LLM_API_KEY=sk-...
SMART_MATCH_LOG_LEVEL=DEBUG
```

---

## Project Structure

```
smart-match/
├── app/
│   ├── api/routes/
│   │   ├── extract.py      # POST /extract
│   │   ├── health.py       # GET /health
│   │   └── results.py      # GET /results/{id}
│   ├── core/
│   │   ├── config.py       # Settings
│   │   ├── llm_client.py   # LLM abstraction
│   │   └── logging.py      # Loguru config
│   ├── services/
│   │   ├── extraction.py   # Info extraction
│   │   ├── ocr.py          # OCR engine
│   │   ├── light_preprocess.py  # Image preprocessing
│   │   └── llm_extraction.py    # LLM extraction
│   └── main.py             # FastAPI app
├── configs/
├── data/
├── results/
├── uploads/
├── scripts/
├── tests/
├── docker-compose.yml
├── Dockerfile
├── Makefile
└── pyproject.toml
```

---

## Development

### Local Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[full]"

# Install Tesseract
sudo apt install tesseract-ocr tesseract-ocr-rus  # Ubuntu
brew install tesseract tesseract-lang              # macOS

python -m app.main
```

### Testing

```bash
pytest
pytest tests/test_extraction.py -v
pytest --cov=app tests/
```

### Code Quality

```bash
ruff check app/
ruff format --check app/
ruff format app/
```

---

## Extraction Methods

| Method       | Condition                   | Description           |
| ------------ | --------------------------- | --------------------- |
| `rule_based` | avg_conf >= 0.7             | Regex + keywords      |
| `llm`        | force_llm or avg_conf < 0.7 | LLM generation        |
| `hybrid`     | Default                     | Rule + LLM validation |

---

## Troubleshooting

### LLM not responding

```bash
curl http://localhost:11434/api/tags
ollama serve
```

### OCR quality issues

- Check input image quality
- Ensure TrOCR model loaded: `make logs | grep TrOCR`
- Try increasing `max_image_dimension`

## Authors

- **Ayomide and Martin**
