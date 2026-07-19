.PHONY: build run stop clean test extract

build:
    docker-compose build

run:
    docker-compose up -d

stop:
    docker-compose down

logs:
    docker-compose logs -f

test:
    curl -s http://localhost:8000/ | python3 -m json.tool

extract:
    curl -s -X POST -F "file=@data/01-0203-0745-001452/00000006.jpg" \
      http://localhost:8000/extract | python3 -m json.tool
