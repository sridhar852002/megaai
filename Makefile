.PHONY: install dev lint typecheck test migrate up down eval

install:
	pip install -e ".[dev]"

dev:
	uvicorn mega_ai.api.app:app --host 0.0.0.0 --port 8000 --reload

lint:
	ruff check mega_ai tests
	ruff format --check mega_ai tests

typecheck:
	mypy mega_ai

test:
	pytest tests -q

migrate:
	alembic upgrade head

eval:
	python scripts/run_eval.py

up:
	docker compose up --build

down:
	docker compose down -v
