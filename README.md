# Smart Match

**AI-powered document intelligence for Russian historical metrical books**

Smart Match extracts structured genealogical data (births, marriages, deaths) from scanned Russian metrical books (метрические книги, XVIII-XX centuries).

---

## Quick Start

```bash
# 1. Start the server
uvicorn app.main:app --port 8000

# 2. Open Swagger docs
open http://localhost:8000/docs

# 3. Extract data from an image
curl -X POST -F "file=@image.jpg" http://localhost:8000/extract
```

### Using Docker

```bash
make build
make run
make test
make docs
```

---

## API Endpoints

| Method | Endpoint        | Description             |
| ------ | --------------- | ----------------------- |
| GET    | `/`             | Service info            |
| GET    | `/health`       | Health check            |
| POST   | `/extract`      | Extract data from image |
| GET    | `/results`      | List all results        |
| GET    | `/results/{id}` | Get result by ID        |

### Example response

```json
{
  "request_id": "a1b2c3d4",
  "file": "metrical_book.jpg",
  "page_type": "table",
  "extracted_data": {
    "record_type": "birth",
    "child_name": { "value": "Иван Петров", "confidence": 0.85 },
    "birth_date": { "value": "1878-03-12", "confidence": 0.9 },
    "father_name": { "value": "Петр Иванов", "confidence": 0.88 },
    "mother_name": { "value": "Анна Иванова", "confidence": 0.86 },
    "needs_review": false
  }
}
```

---

## Makefile Commands

```bash
make help           Show help
make build          Build Docker image
make run            Start container
make stop           Stop container
make logs           Show logs
make test           Test API
make health         Check health
make extract        Extract from sample image
make results        List results
make docs           Open Swagger
make shell          Enter container
make status         Container status
```

---

## Pipeline Architecture

```bash
Input Image
    ↓
[1] Light Preprocessing (deskew, color/illumination normalize, CLAHE, denoise)
    ↓
[2] Layout Detection (table vs text, cell extraction)
    ↓
[3] OCR (EasyOCR Russian + Tesseract fallback)
    ↓
[4] Post-processing (spelling, abbreviations, pre-1918 orthography)
    ↓
[5] Information Extraction (rule-based: birth/death/marriage)
    ↓
[6] Confidence Scoring + needs_review flag
    ↓
Structured JSON Output
```

---

## Project Structure

```bash
smart-match/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── api/routes/          # API endpoints
│   ├── services/            # Business logic
│   │   ├── light_preprocess.py    # Image preprocessing
│   │   ├── layout.py              # Table/text detection
│   │   ├── ocr.py                 # EasyOCR + Tesseract
│   │   ├── region_preprocess.py   # Region-specific preprocessing
│   │   ├── postprocessing.py      # Text correction
│   │   ├── extraction.py          # Field extraction
│   │   └── llm_extraction.py      # LLM fallback
│   └── schemas/             # Pydantic models
├── configs/                 # Configuration files
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── README.md
```

---

## Tech Stack

| Component        | Technology             |
| ---------------- | ---------------------- |
| API              | FastAPI, Pydantic      |
| OCR              | EasyOCR, Tesseract     |
| Image Processing | OpenCV, NumPy          |
| Logging          | Loguru                 |
| Container        | Docker, Docker Compose |
| Language         | Python 3.11+           |

---

## Development

```bash
# Install dependencies
pip install -e .

# Run tests
pytest tests/ -v

# Run locally
uvicorn app.main:app --reload --port 8000
```

---

## Team

- **Ayomide Isreal Ajibade** — a.ajibade@innopolis.university
- **Niyonshuti Martin** — m.niyonshuti@innopolis.university

Innopolis University, 2026
