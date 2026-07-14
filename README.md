# Smart Match

**Intelligent Information Extraction from Russian Historical Metrical Books**

**Smart Match** is an AI-powered system that automatically extracts structured genealogical information from scanned Russian historical metrical books (метрические книги — records of births, marriages, and deaths). It processes scanned images, recognizes handwritten and printed Russian text (including pre-1918 orthography), extracts relevant fields, normalizes names and dates, and returns structured JSON — all through a simple REST API.

Built for Russian speakers researching genealogical data from the Russian Empire, Congress Poland, and Soviet-era records.

---

## Key Features

- **Russian-optimized OCR** — Uses `taiga75/ru-trocr-1700s` fine-tuned on 18th-19th century Russian Civil Script
- **Pre-1918 orthography support** — Handles ѣ, і, ѳ, ѵ, ъ (yat, decimal i, fita, izhitsa, hard sign)
- **Roman numeral dates** — Supports `12.III.1878` format common in Russian records
- **Orthodox calendar** — Resolves Пасха (Easter), Рождество, Троица and other movable feasts
- **Russian name variants** — Normalizes Иван → Иоанн → Иван, including patronymics
- **Tesseract fallback** — Uses Tesseract with `rus` language for clean printed text
- **LLM-assisted extraction** — Falls back to Ollama (Llama 3.1 8B) with Russian prompts for ambiguous cases
- **Confidence scoring** — Every field has a reliability score; low-confidence fields flagged for review

---

## Project Structure

```bash
smart-match/
├── app/                          # Main application package
│   ├── api/                      # FastAPI routes
│   │   ├── routes/
│   │   │   ├── health.py         # Health check endpoint
│   │   │   └── extract.py        # Document extraction endpoint (lang: ru)
│   │   └── __init__.py
│   ├── core/                     # Configuration & logging
│   │   ├── config.py             # Pydantic settings (env vars, validation)
│   │   └── logging.py            # Loguru setup with rotation
│   ├── main.py                   # FastAPI app entry point
│   ├── schemas/                  # Pydantic data models
│   │   ├── common.py             # FieldPrediction, NameResolution
│   │   ├── birth.py              # Birth record schema
│   │   ├── death.py              # Death record schema
│   │   └── marriage.py           # Marriage record schema
│   ├── services/                 # Business logic
│   │   ├── preprocessing.py      # Image enhancement (OpenCV)
│   │   ├── ocr.py                # ru-trocr-1700s + Tesseract rus
│   │   ├── postprocessing.py     # Russian spelling/abbreviation correction
│   │   ├── extraction.py         # Russian field extraction (regex)
│   │   ├── llm_extraction.py     # LLM fallback for ambiguous cases
│   │   └── entity_resolution.py  # Russian names + Orthodox dates
│   └── utils/
│       └── file_utils.py         # File I/O helpers
├── configs/                      # Environment-specific configs
│   ├── config.yml                # Default configuration
│   ├── config.dev.yml            # Development overrides
│   ├── config.staging.yml        # Staging overrides
│   └── config.prod.yml           # Production overrides
├── data/
│   ├── input/                    # Uploaded images
│   ├── output/                   # Extracted JSON results
│   ├── cache/                    # OCR result cache
│   └── samples/                  # Sample test images
├── docker/                       # Docker helper files
├── docs/                         # Documentation
├── logs/                         # Application logs (gitignored)
├── configs/                      # Environment configs
├── .gitignore
├── docker-compose.yml            # Docker Compose service definition
├── Dockerfile                    # Container build instructions
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Conda](https://docs.conda.io/) (recommended) or venv
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) with Russian language pack
- [Docker](https://www.docker.com/) (for containerized deployment)
- 4GB+ RAM (for TrOCR model)
- Optional: [Ollama](https://ollama.ai/) with `llama3.1:8b` for LLM-assisted extraction

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/smart-match.git
cd smart-match

# 2. Create and activate conda environment
conda create -n ai_sec python=3.11 -y
conda activate ai_sec

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Tesseract with Russian language
brew install tesseract tesseract-lang  # macOS
# sudo apt install tesseract-ocr tesseract-ocr-rus  # Linux

# 5. Install Ollama for LLM extraction
# curl -fsSL https://ollama.ai/install.sh | sh
# ollama pull llama3.1:8b

# 6. Run the server
uvicorn app.main:app --reload
```

The server starts at **http://localhost:8000**.

### Using Docker

```bash
# Build and run
docker-compose up --build
```

---

## API Reference

### `GET /`

Root endpoint — returns service status.

```json
{
  "service": "Smart Match",
  "status": "running"
}
```

### `GET /health`

Health check endpoint — used for liveness probes.

```json
{
  "status": "healthy",
  "service": "smart-match"
}
```

### `POST /extract`

Upload a scanned Russian metrical book page for processing.

**Request:**

- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` — image file (JPG, PNG, TIFF, BMP; max 10MB)
- Query: `?language=ru` (optional, defaults to `ru`)

**Response (success):**

```json
{
  "filename": "metrical_book_1878.jpg",
  "status": "processed",
  "language": "ru",
  "ocr_engine": "ru-trocr-1700s",
  "ocr_confidence": 0.85,
  "post_processing": ["spelling_correction", "abbreviations_expanded"],
  "record": {
    "record_type": "birth",
    "child_name": {
      "value": "Иван Петров",
      "confidence": 0.85
    },
    "birth_date": {
      "value": "1878-03-12",
      "confidence": 0.9
    },
    "father_name": {
      "value": "Петр Иванов",
      "confidence": 0.88
    },
    "mother_name": {
      "value": "Анна Иванова",
      "confidence": 0.86
    },
    "needs_review": false,
    "child_name_resolved": {
      "original": "Иван Петров",
      "canonical": "Иван Петров",
      "variants": ["Иван", "Иоанн", "Ivan", "Ваня"],
      "first_name": "Иван",
      "last_name": "Петров",
      "confidence": 0.9
    }
  }
}
```

### Error Handling

| Status Code | Meaning                                     |
| ----------- | ------------------------------------------- |
| 400         | Invalid file type or file too large         |
| 422         | Image could not be decoded or no text found |
| 500         | Processing pipeline error                   |

---

## Example Usage

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test root endpoint
curl http://localhost:8000/

# Extract from a Russian metrical book page
curl -X POST \
  -F "file=@metrical_book_1878.jpg" \
  http://localhost:8000/extract
```

---

## Pipeline Architecture

```
Input Image (Russian metrical book scan)
     ↓
┌─ Preprocessing ──────────────────────┐
│ • Grayscale conversion               │
│ • CLAHE contrast enhancement         │
│ • Deskew (Hough line detection)      │
│ • Border removal & margin cropping   │
│ • Image size limiting (max 4000px)   │
└──────────────────────────────────────┘
     ↓
┌─ OCR Engine ─────────────────────────┐
│ Primary: ru-trocr-1700s              │
│   (Russian 18th-century Civil Script)│
│ Fallback: Tesseract with rus         │
│ Voting: picks best confidence result │
└──────────────────────────────────────┘
     ↓
┌─ Post-Processing ────────────────────┐
│ • Russian spelling correction        │
│ • Pre-1918 orthography (ѣ→е, і→и)    │
│ • Abbreviation expansion (р.→родился)│
│ • Date normalization (12.III.1878)   │
└──────────────────────────────────────┘
     ↓
┌─ Information Extraction ─────────────┐
│ Rule-based: Russian regex patterns   │
│   (родился, умер, венчался)          │
│ LLM fallback: Ollama Llama 3.1 8B    │
│   (when confidence < 0.7)            │
└──────────────────────────────────────┘
     ↓
┌─ Entity Resolution ──────────────────┐
│ • Name normalization (Иван→Иоанн)    │
│ • Orthodox calendar (Пасха, Троица)  │
│ • Age validation from dates          │
│ • Family linking across records      │
└──────────────────────────────────────┘
     ↓
Structured JSON Output
```

---

## Models Used

| Model              | Source                                                       | Purpose                            | CER   |
| ------------------ | ------------------------------------------------------------ | ---------------------------------- | ----- |
| **ru-trocr-1700s** | [HuggingFace](https://huggingface.co/taiga75/ru-trocr-1700s) | Russian Civil Script HTR           | 1.69% |
| **Llama 3.1 8B**   | [Ollama](https://ollama.ai/)                                 | LLM-assisted extraction (optional) | -     |
| **Tesseract rus**  | Local install                                                | Printed text fallback              | -     |

---

## Current Development Status

| Task                                        | Status                 | Completion      |
| ------------------------------------------- | ---------------------- | --------------- |
| **Task 1:** System Design & Setup           | Complete               | 12/12           |
| **Task 2:** Image Preprocessing             | Complete               | 12/12           |
| **Task 3:** Layout Detection                | Not Started (deferred) | 0/12            |
| **Task 4:** OCR Pipeline                    | Complete               | 12/12           |
| **Task 5:** Information Extraction          | Complete               | 12/12           |
| **Task 6:** Data Normalization & Confidence | In Progress            | 3/12            |
| **Task 7:** API, Docker & Deployment        | In Progress            | 4/12            |
| **Total**                                   |                        | **55/84 (65%)** |

---

## Technology Stack

| Component            | Technology                     |
| -------------------- | ------------------------------ |
| **Web Framework**    | FastAPI                        |
| **Data Validation**  | Pydantic, pydantic-settings    |
| **Image Processing** | OpenCV, NumPy, Pillow          |
| **Russian OCR**      | taiga75/ru-trocr-1700s (TrOCR) |
| **Fallback OCR**     | Tesseract 5 + rus.traineddata  |
| **LLM**              | Ollama + Llama 3.1 8B          |
| **Deep Learning**    | PyTorch (MPS/CUDA)             |
| **Logging**          | Loguru                         |
| **Containerization** | Docker, Docker Compose         |
| **Language**         | Python 3.11                    |

---

## Dependencies

Key Python packages:

```
torch>=2.0.0
transformers>=4.30.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
loguru>=0.7.0
httpx>=0.25.0
```

---

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

- Follow PEP 8
- Type hints required for all functions
- Docstrings for all modules and public functions

### Adding Dependencies

```bash
pip install <package>
pip freeze | grep <package> >> requirements.txt
```

---

## License

This project is for educational and research purposes.

---

## Acknowledgments

- **Maria Levchenko** — Creator of `taiga75/ru-trocr-1700s` model
- **READ-COOP** — Transkribus public Russian models
- **Microsoft TrOCR team** — Base transformer architecture
- **FastAPI team** — Web framework
- **Ollama team** — Local LLM inference

---

_Built with ❤️ for Russian genealogical research_
