.PHONY: install lint format test clean run-train run-predict

install:
	poetry install

lint:
	poetry run ruff check src/ tests/
	poetry run mypy src/
	poetry run bandit -c pyproject.toml -r src/

format:
	poetry run ruff format src/ tests/
	poetry run ruff check --fix src/ tests/

test:
	poetry run pytest -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -f .coverage

run-train:
	poetry run python -m price_prediction.models.train

run-predict:
	poetry run python test_predict.py

jupyter:
	poetry run jupyter notebook
