.PHONY: install dev test test-integration lint format build push clean load-test chaos-test

install:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

dev:
	docker-compose up -d
	uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/unit/ -v --cov=app --cov=training --cov=index --cov=streaming --cov=feature_store --cov-report=term-missing

test-integration:
	docker-compose -f docker-compose.test.yml up -d
	pytest tests/integration/ -v -m integration
	docker-compose -f docker-compose.test.yml down

lint:
	black --check --line-length 100 .
	flake8 --max-line-length 100 --extend-ignore E203,W503 app/ training/ index/ streaming/ feature_store/
	mypy --ignore-missing-imports app/ training/ index/ streaming/ feature_store/

format:
	black --line-length 100 .
	isort .

build:
	docker build -t rec-engine:latest -f infrastructure/docker/Dockerfile .

push:
	docker push rec-engine:latest

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage

load-test:
	locust -f load_testing/locustfile.py --host http://localhost:8000 --users 100 --spawn-rate 10 --run-time 60s

chaos-test:
	python load_testing/chaos_testing.py --experiment pod_deletion --duration 60s
