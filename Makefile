.PHONY: help build run stop restart clean logs test extract extract-small results health docs shell status

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
	@echo "  make fmt            Formating and checking"
	@echo ""
	@echo "Quick start: make build run test"

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
	curl -s http://localhost:8000/ | python3 -m json.tool

health:
	curl -s http://localhost:8000/health | python3 -m json.tool

extract:
	curl -s -X POST \
	  -F "file=@data/01-0203-0745-001452/00000006.jpg" \
	  http://localhost:8000/extract | python3 -m json.tool

extract-small:
	curl -s -X POST \
	  -F "file=@data/01-0203-0745-000600/00000009.jpg" \
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
