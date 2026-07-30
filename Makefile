.PHONY: help build run stop restart clean logs test extract extract-small results health docs shell status lint format fmt

help:
	@echo "Smart Match API — Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make help           Show this help message"
	@echo "  make build          Build Docker image"
	@echo "  make run            Start Docker container"
	@echo "  make stop           Stop Docker container"
	@echo "  make restart        Restart Docker container"
	@echo "  make logs           Show container logs"
	@echo "  make clean          Remove container and prune system"
	@echo "  make test           Test API root endpoint"
	@echo "  make health         Test API health endpoint"
	@echo "  make extract        Test extraction on a sample image"
	@echo "  make extract-small  Test extraction on a small image"
	@echo "  make results        List all extraction results"
	@echo "  make docs           Open Swagger documentation"
	@echo "  make shell          Open shell inside container"
	@echo "  make status         Show container status"
	@echo "  make lint           Check code for errors"
	@echo "  make format         Sorting codes"
	@echo "  make fmt            Format and check"

build:
	docker compose build

run:
	docker compose up -d

stop:
	docker compose down

restart: stop run

logs:
	docker compose logs -f

clean:
	docker compose down -v
	docker system prune -f

test:
	@echo "Testing API..."
	@for i in $$(seq 1 15); do \
	if curl -sf http://localhost:8000/ > /dev/null 2>&1; then \
	curl -s http://localhost:8000/ | python3 -m json.tool; \
	exit 0; \
	fi; \
	echo "  Waiting for API ($$i/15)..."; \
	sleep 2; \
	done; \
	echo '{"error": "API did not start in time"}'

health:
	@echo "Checking health..."
	@for i in $$(seq 1 5); do \
	result=$$(curl -sf http://localhost:8000/health 2>/dev/null); \
	if [ -n "$$result" ]; then \
	echo "$$result" | python3 -m json.tool; \
	exit 0; \
	fi; \
	sleep 2; \
	done; \
	echo '{"error": "Health endpoint unavailable"}'

extract:
	curl -s -X POST \
	-F "file=@data/images/00000006.jpg" \
	http://localhost:8000/extract | python3 -m json.tool

extract-small:
	curl -s -X POST \
	-F "file=@data/images/00000009.jpg" \
	http://localhost:8000/extract | python3 -m json.tool

results:
	curl -s http://localhost:8000/results/ | python3 -m json.tool

docs:
	open http://localhost:8000/docs

shell:
	docker compose exec smart-match /bin/bash

status:
	docker compose ps

lint:
	ruff check app/ tests/

format:
	ruff format app/ tests/

fmt: format lint
