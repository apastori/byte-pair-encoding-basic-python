# ======================================================
#  Python Project Makefile
#  Tools: Black (formatter), Ruff (linter), MyPy (type checker)
# ======================================================

.PHONY: help format format-check lint lint-fix typecheck qa clean install
.PHONY: run-tests-init

# Default target - show help
help:
	@echo "Available commands:"
	@echo "  make format        - Format code with Black"
	@echo "  make format-check  - Check formatting without modifying files"
	@echo "  make lint          - Lint code with Ruff (no fixes)"
	@echo "  make lint-fix      - Lint code with Ruff and auto-fix issues"
	@echo "  make typecheck     - Type check code with MyPy"
	@echo "  make qa            - Run all quality checks (format + lint-fix + typecheck)"
	@echo "  make clean         - Remove cache directories and temporary files"
	@echo "  make install       - Install development dependencies"

# Format code with Black
format:
	@echo "🎨 Formatting code with Black..."
	black .
	@echo "✅ Formatting complete!"

# Check if code is formatted (CI/CD friendly)
format-check:
	@echo "🔍 Checking code formatting..."
	black . --check --diff

# Lint with Ruff (check only, no fixes)
lint:
	@echo "🔍 Linting code with Ruff..."
	ruff check .

# Lint with Ruff and auto-fix issues
lint-fix:
	@echo "🔧 Linting and fixing code with Ruff..."
	ruff check . --fix
	@echo "✅ Linting complete!"

# Type check with MyPy
typecheck:
	@echo "🔬 Type checking with MyPy..."
	mypy .
	@echo "✅ Type checking complete!"

# Run all quality assurance checks
qa: format lint-fix typecheck
	@echo ""
	@echo "✅ All quality checks passed!"

# Clean cache directories and temporary files
clean:
	@echo "🧹 Cleaning cache directories..."
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.mypy_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.ruff_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	find . -type f -name '*.pyo' -delete 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# Install development dependencies
install:
	@echo "📦 Installing development dependencies..."
	pip install -r requirements.txt
	@echo "✅ Installation complete!"

# Run the test package __init__.py directly (for quick ad-hoc testing)
run-tests-init:
	@echo "▶ Running src/tests/__init__.py..."
	python -u src/tests/__init__.py
	@echo "✅ Run complete"