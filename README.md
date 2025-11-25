# Byte-Pair Encoding

This project provides a basic implementation of the Byte-Pair Encoding (BPE) algorithm in Python. BPE is a data compression technique that iteratively replaces the most frequent pair of bytes in a sequence with a new byte that was not present in the original data. This implementation is designed to be simple and easy to understand, making it a good starting point for learning about BPE.

This project is also a basic Python project boilerplate that includes a set of tools to ensure code quality and consistency. It comes pre-configured with Black for code formatting, Ruff for linting, and MyPy for static type checking.

## Directory Structure

The project is organized as follows:

```
.
├── .github/                # GitHub Actions workflows
├── .venv/                  # Virtual environment
├── data/                   # Data files for training and validation
├── docs/                   # Project documentation
├── notebooks/              # Jupyter notebooks for experimentation
├── src/                    # Source code
│   ├── minbpe/             # Main package for the BPE implementation
│   │   ├── __init__.py     # Package initializer
│   │   ├── basic.py        # Basic BPE implementation
│   │   └── regex.py        # BPE implementation with regex
│   └── tests/              # Tests for the BPE implementation
│       ├── __init__.py     # Test package initializer
│       └── test_basic.py   # Tests for the basic BPE implementation
├── .gitignore              # Files to ignore in Git
├── .python-version         # Python version for pyenv
├── LICENSE                 # Project license
├── Makefile                # Makefile with common commands
├── pyproject.toml          # Project metadata and dependencies
├── README.md               # This file
├── requirements.txt        # Project dependencies
├── runtime.txt             # Python runtime version for Heroku
└── tasks.py                # Invoke tasks for automation
```

## Tooling

This boilerplate uses the following tools to maintain code quality:

### Black

[Black](https://github.com/psf/black) is the uncompromising Python code formatter. It enforces a strict and consistent code style, which helps to eliminate debates over formatting and makes code review faster. By using Black, you ensure that the code looks the same regardless of who wrote it.

### Ruff

[Ruff](https.github.com/astral-sh/ruff) is an extremely fast Python linter, written in Rust. It can check for a wide range of errors and style issues, from simple syntax mistakes to complex bugs. It also includes a code formatter. In this boilerplate, it is used for linting and fixing common issues automatically.

### MyPy

[MyPy](http.mypy-lang.org/) is a static type checker for Python. It helps you write cleaner, more robust code by adding type hints to your functions and variables. MyPy can catch type-related errors before you even run your code, which can save you a lot of time debugging.

## Automation

This project uses `make` and `invoke` to automate common development tasks.

### Makefile

A `Makefile` is included to provide a simple and familiar interface for common commands. You can see the available commands by running `make help`.

### Invoke & tasks.py

[Invoke](https.www.pyinvoke.org/) is a Python task execution tool. The `tasks.py` file defines a set of tasks that can be run from the command line. This allows for more complex logic than a `Makefile` and is written in Python. You can see the available tasks by running `invoke --list`.

## Available Commands

You can use either `make` or `invoke` to run the following commands:

| Command | Makefile | Invoke | Description |
|---|---|---|---|
| **Format Code** | `make format` | `invoke format` | Formats the code using Black. |
| **Check Formatting** | `make format-check` | `invoke format-check` | Checks if the code is formatted correctly without making changes. |
| **Lint Code** | `make lint` | `invoke lint` | Lints the code with Ruff to find potential issues. |
| **Lint and Fix** | `make lint-fix` | `invoke lint-fix` | Lints the code with Ruff and automatically fixes any issues it can. |
| **Type Check** | `make typecheck` | `invoke typecheck` | Runs MyPy to check for type errors. |
| **Quality Assurance** | `make qa` | `invoke qa` | Runs all the quality checks: format, lint-fix, and typecheck. |
| **Install Dependencies**| `make install` | `invoke install` | Installs the required Python packages from `requirements.txt`. |
| **Clean Project** | `make clean` | `invoke clean` | Removes cache directories and temporary files. |
| **Help** | `make help` | `invoke help` | Shows the list of available commands. |
