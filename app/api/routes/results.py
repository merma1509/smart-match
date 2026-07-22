import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/results", tags=["results"])
RESULTS_DIR = Path("results")


@router.get("/")
def list_results():
    """Get list of all extraction results."""
    if not RESULTS_DIR.exists():
        return {"results": []}

    results = []
    for f in sorted(RESULTS_DIR.glob("*.json"), reverse=True)[:50]:
        try:
            with open(f) as fp:
                data = json.load(fp)
                results.append(
                    {
                        "id": data.get("request_id", f.stem),
                        "file": data.get("file", "unknown"),
                        "type": data.get("extracted_data", {}).get("record_type", "unknown"),
                        "needs_review": data.get("needs_review", True),
                        "time": data.get("processing_time_seconds", 0),
                    }
                )
        except Exception:
            pass

    return {"total": len(results), "results": results}


@router.get("/{result_id}")
def get_result(result_id: str):
    """Get a specific result by ID."""
    result_path = RESULTS_DIR / f"{result_id}.json"
    if not result_path.exists():
        # Try without extension
        for f in RESULTS_DIR.glob(f"{result_id}*.json"):
            result_path = f
            break

    if not result_path.exists():
        raise HTTPException(404, f"Result {result_id} not found")

    with open(result_path) as f:
        return json.load(f)
