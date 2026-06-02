# Contributing to Real-Time Recommendation Engine

Thank you for your interest in contributing to the Real-Time Recommendation Engine! We welcome contributions from the community and are grateful for every pull request, bug report, and feature suggestion.

---

## 📜 Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. We are committed to providing a welcoming and inclusive environment for everyone. Please be respectful, constructive, and professional in all interactions.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed on your machine:

- **Python** 3.10+
- **Docker** & **Docker Compose** v2+
- **Git** 2.30+
- **Make** (optional, for convenience targets)

### Setting Up Your Development Environment

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/<your-username>/Real-time-recommendation-engine.git
   cd Real-time-recommendation-engine/realtime-rec-engine-v2
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate    # macOS / Linux
   # .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -r requirements-dev.txt   # if available
   ```

4. **Start infrastructure services**

   ```bash
   docker compose up -d
   ```

5. **Run the test suite**

   ```bash
   pytest --tb=short -q
   ```

   All tests should pass before you start making changes.

---

## 🔄 Development Workflow

### Branch Naming Convention

Create a descriptive branch from `main` using one of the following prefixes:

| Prefix      | Purpose                          | Example                          |
|-------------|----------------------------------|----------------------------------|
| `feature/`  | New features                     | `feature/user-embedding-v2`      |
| `fix/`      | Bug fixes                        | `fix/cache-invalidation-race`    |
| `docs/`     | Documentation updates            | `docs/api-endpoint-guide`        |
| `refactor/` | Code restructuring               | `refactor/streaming-pipeline`    |
| `test/`     | Adding or updating tests         | `test/ab-testing-integration`    |

```bash
git checkout -b feature/your-feature-name
```

### Conventional Commits

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification. Every commit message **must** use one of the following prefixes:

| Prefix   | When to use                                      |
|----------|--------------------------------------------------|
| `feat:`  | A new feature                                    |
| `fix:`   | A bug fix                                        |
| `docs:`  | Documentation-only changes                       |
| `test:`  | Adding or correcting tests                       |
| `chore:` | Maintenance tasks (CI, deps, tooling)            |
| `refactor:` | Code change that neither fixes a bug nor adds a feature |
| `perf:`  | Performance improvement                          |

**Examples:**

```bash
git commit -m "feat: add real-time feature store integration"
git commit -m "fix: resolve race condition in cache invalidation"
git commit -m "docs: update API reference for /recommend endpoint"
git commit -m "test: add integration tests for Kafka consumer"
git commit -m "chore: bump prometheus-client to 0.19.0"
```

---

## 📋 Pull Request Process

1. **Describe your changes** clearly in the PR description, including the motivation and context.
2. **Link related issues** using `Closes #123` or `Relates to #456`.
3. **Ensure CI passes** — all checks (lint, type-check, tests, security scan) must be green.
4. **Request at least 1 review** from a maintainer or team member.
5. **Keep PRs focused** — avoid mixing unrelated changes in a single PR.
6. **Update documentation** if your change affects public APIs or configuration.

### PR Checklist

```markdown
- [ ] Code follows the project's style guidelines
- [ ] Self-review of the code has been performed
- [ ] Tests have been added/updated for the changes
- [ ] Documentation has been updated (if applicable)
- [ ] All CI checks pass
- [ ] No new warnings are introduced
```

---

## ✅ Code Quality Standards

We maintain high code quality through automated tooling. Please ensure your code passes all checks before submitting a PR.

### Formatting & Linting

```bash
# Code formatting (line length = 100)
black --line-length 100 --check .

# Apply formatting
black --line-length 100 .

# Linting
flake8 --max-line-length 100 --exclude .venv,__pycache__
```

### Type Checking

```bash
mypy app/ --ignore-missing-imports
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=term-missing

# Run only unit tests
pytest tests/unit/

# Run only integration tests (requires Docker services)
pytest tests/integration/
```

### Security Scanning

```bash
bandit -r app/ -ll
```

### Summary of Standards

| Tool      | Purpose              | Configuration                    |
|-----------|----------------------|----------------------------------|
| `black`   | Code formatting      | `--line-length 100`              |
| `flake8`  | Linting              | `--max-line-length 100`          |
| `mypy`    | Static type checking | `--ignore-missing-imports`       |
| `pytest`  | Testing framework    | `--cov=app`                      |
| `bandit`  | Security analysis    | `-r app/ -ll`                    |

---

## 🐛 Reporting Bugs

If you find a bug, please open an issue with the following information:

1. **Summary** — A clear and concise description of the bug.
2. **Steps to Reproduce** — Detailed steps to reproduce the behavior:
   1. Set up environment with '...'
   2. Run command '...'
   3. Observe error at '...'
3. **Expected Behavior** — What you expected to happen.
4. **Actual Behavior** — What actually happened, including error messages and stack traces.
5. **Environment** — Include relevant details:
   - OS and version (e.g., Ubuntu 22.04, macOS 14.1)
   - Python version (`python --version`)
   - Docker version (`docker --version`)
   - Relevant dependency versions
6. **Screenshots / Logs** — Attach any relevant screenshots or log output.

---

## 💡 Requesting Features

We love hearing ideas! To suggest a new feature:

1. **Check existing issues** to see if the feature has already been requested.
2. **Open a new issue** with the `enhancement` label and include:
   - **Problem Statement** — What problem does this feature solve?
   - **Proposed Solution** — How do you envision this working?
   - **Alternatives Considered** — Any alternative approaches you've thought about.
   - **Additional Context** — Mockups, references, or examples from other projects.

---

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

Thank you for helping make the Real-Time Recommendation Engine better! 🎉
