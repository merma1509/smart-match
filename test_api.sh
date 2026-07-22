#!/bin/bash
BASE="http://localhost:8000"
PASS=0
FAIL=0

echo "=== Smart Match Tests ==="
echo ""

# 1. Root
echo "1. GET /"
RESP=$(curl -s $BASE/)
if echo "$RESP" | grep -q "Smart Match"; then
  echo "   [COMPLETED] Root endpoint works"
  ((PASS++))
else
  echo "   [FAILED] Root endpoint failed"
  ((FAIL++))
fi

# 2. Health
echo "2. GET /health"
RESP=$(curl -s $BASE/health)
if echo "$RESP" | grep -q "healthy"; then
  echo "  [COMPLETED] Health check passed"
  ((PASS++))
else
  echo "   [FAILED] Health check failed"
  ((FAIL++))
fi

# 3. Create test image if needed
if [ ! -f "test_input.jpg" ]; then
  python3 -c "
import cv2, numpy as np
img = np.ones((200, 800, 3), dtype=np.uint8) * 255
cv2.putText(img, 'Test metrical book 1890', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
cv2.imwrite('test_input.jpg', img)
" 2>/dev/null
fi

# 4. Extract
echo "3. POST /extract"
RESP=$(curl -s -X POST $BASE/extract -F "file=@test_input.jpg")
ID=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('request_id','?'))" 2>/dev/null)
if [ "$ID" != "?" ]; then
  echo "  [COMPLETED] Extract endpoint works (ID: $ID)"
  ((PASS++))
else
  echo "   [FAILED] Extract endpoint failed"
  echo "$RESP"
  ((FAIL++))
fi

# 5. Batch
echo "4. POST /extract/batch"
RESP=$(curl -s -X POST $BASE/extract/batch -F "files=@test_input.jpg" -F "files=@test_input.jpg")
TOTAL=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total',0))" 2>/dev/null)
if [ "$TOTAL" = "2" ]; then
  echo "  [COMPLETED] Batch endpoint works (processed $TOTAL)"
  ((PASS++))
else
  echo "   [FAILED] Batch endpoint failed"
  echo "$RESP"
  ((FAIL++))
fi

# 6. List results
echo "5. GET /results"
RESP=$(curl -s "$BASE/results/?limit=5&offset=0")
TOTAL=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total','?'))" 2>/dev/null)
if [ "$TOTAL" != "?" ]; then
  echo "  [COMPLETED] List results works (total: $TOTAL)"
  ((PASS++))
else
  echo "   [FAILED] List results failed"
  ((FAIL++))
fi

# 7. Get specific result
echo "6. GET /results/$ID"
RESP=$(curl -s "$BASE/results/$ID")
if echo "$RESP" | grep -q "request_id"; then
  echo "  [COMPLETED] Get result by ID works"
  ((PASS++))
else
  echo "   [FAILED] Get result by ID failed"
  ((FAIL++))
fi

# 8. Delete result
echo "7. DELETE /results/$ID"
RESP=$(curl -s -X DELETE "$BASE/results/$ID")
if echo "$RESP" | grep -q "deleted"; then
  echo "  [COMPLETED] Delete result works"
  ((PASS++))
else
  echo "   [FAILED] Delete result failed"
  ((FAIL++))
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="