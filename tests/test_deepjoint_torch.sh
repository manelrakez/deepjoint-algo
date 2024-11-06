set -eux

#python -m ruff check src tests || true
#python -m ruff check src tests --fix --fix-only
#python -m ruff format src tests --check --verbose

pytest --capture=no
