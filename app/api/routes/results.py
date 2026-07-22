"""Results endpoints — list, get, delete extraction results."""
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/results", tags=["results"])

RESULTS_DIR = Path("results")


@router.get("/")
def list_results(
    limit: int = Query(20, ge=1, le=100, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
):
    """Get list of all extraction results with pagination."""
    if not RESULTS_DIR.exists():
        return {"total": 0, "limit": limit, "offset": offset, "results": []}

    all_files = sorted(RESULTS_DIR.glob("*.json"), reverse=True)
    total = len(all_files)
    page = all_files[offset : offset + limit]

    results = []
    for f in page:
        try:
            with open(f, encoding="utf-8") as fp:
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

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "results": results,
    }


@router.get("/{result_id}")
def get_result(result_id: str):
    """Get a specific result by ID."""
    result_path = RESULTS_DIR / f"{result_id}.json"
    if not result_path.exists():
        for f in RESULTS_DIR.glob(f"{result_id}.*"):
            result_path = f
            break

    if not result_path.exists():
        raise HTTPException(404, f"Result {result_id} not found")

    with open(result_path, encoding="utf-8") as f:
        return json.load(f)


@router.delete("/{result_id}")
def delete_result(result_id: str):
    """Delete a specific result by ID."""
    result_path = RESULTS_DIR / f"{result_id}.json"
    if not result_path.exists():
        raise HTTPException(404, f"Result {result_id} not found")

    # Also delete the uploaded file
    for ext in [".jpg", ".jpeg", ".png"]:
        upload_path = Path("uploads") / f"{result_id}{ext}"
        if upload_path.exists():
            upload_path.unlink()
            break

    result_path.unlink()
    return {"status": "deleted", "id": result_id}
