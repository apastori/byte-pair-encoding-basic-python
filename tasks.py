"""Task runner for Python project using Invoke.

Installation:
    pip install invoke

Usage:
    invoke help            - Show available tasks
    invoke format          - Format code with Black
    invoke lint            - Lint code with Ruff
    invoke lint-fix        - Lint and auto-fix with Ruff
    invoke typecheck       - Type check with MyPy
    invoke qa              - Run all quality checks
    invoke clean           - Clean cache directories
    invoke install         - Install development dependencies
    invoke check-all       - Run all checks without modifying files
    invoke --list          - Show all available tasks
"""

import shutil
import sys
from pathlib import Path

from invoke.context import Context
from invoke.tasks import task


@task
def help(c: Context) -> None:
    """Show available tasks."""
    c.run("invoke --list")


@task
def format(c: Context) -> None:
    """Format code with Black."""
    print("🎨 Formatting code with Black...")
    c.run("black .")
    print("✅ Formatting complete!")


@task
def format_check(c: Context) -> None:
    """Check code formatting without modifying files."""
    print("🔍 Checking code formatting...")
    c.run("black . --check --diff")


@task
def lint(c: Context) -> None:
    """Lint code with Ruff (no fixes)."""
    print("🔍 Linting code with Ruff...")
    c.run("ruff check .")


@task
def lint_fix(c: Context) -> None:
    """Lint code with Ruff and auto-fix issues."""
    print("🔧 Linting and fixing code with Ruff...")
    c.run("ruff check . --fix")
    print("✅ Linting complete!")


@task
def typecheck(c: Context) -> None:
    """Type check code with MyPy."""
    print("🔬 Type checking with MyPy...")
    c.run("mypy .")
    print("✅ Type checking complete!")


@task(pre=[format, lint_fix, typecheck])
def qa(c: Context) -> None:
    """Run all quality assurance checks."""
    print("\n" + "=" * 50)
    print("✅ All quality checks passed!")
    print("=" * 50)


@task
def clean(c: Context) -> None:
    """Remove cache directories and temporary files."""
    print("🧹 Cleaning cache directories...")

    cache_dirs = [
        '__pycache__',
        '.mypy_cache',
        '.ruff_cache',
        '.pytest_cache',
        '*.egg-info',
    ]

    cache_files = [
        '*.pyc',
        '*.pyo',
        '*.pyd',
    ]

    # Remove directories
    for pattern in cache_dirs:
        for path in Path('.').rglob(pattern):
            if path.is_dir():
                print(f"  Removing {path}")
                shutil.rmtree(path, ignore_errors=True)

    # Remove files
    for pattern in cache_files:
        for path in Path('.').rglob(pattern):
            if path.is_file():
                print(f"  Removing {path}")
                path.unlink(missing_ok=True)

    print("✅ Cleanup complete!")


@task
def install(c: Context) -> None:
    """Install development dependencies."""
    print("📦 Installing development dependencies...")
    c.run(f"{sys.executable} -m pip install -r requirements.txt")
    print("✅ Installation complete!")


@task
def check_all(c: Context) -> None:
    """Run format check, lint, and typecheck (for CI/CD)."""
    print("🔍 Running all checks without modifications...")
    format_check(c)
    lint(c)
    typecheck(c)
    print("\n✅ All checks passed!")


@task
def run_tests_init(c: Context) -> None:
    """Run the `src/tests/__init__.py` file directly for ad-hoc testing.

    This runs the file as a script using the current Python interpreter.
    Useful for quickly exercising any module-level test code placed in
    `src/tests/__init__.py` without running the full test suite.
    """
    target = Path("src/tests/__init__.py")
    if not target.exists():
        print(f"Error: {target} not found. Nothing to run.")
        return
    print(f"▶ Running {target} with {sys.executable}...")
    c.run(f"{sys.executable} -u {target}")
    print("✅ Finished running tests init file")
