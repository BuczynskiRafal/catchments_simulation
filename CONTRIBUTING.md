# Contributing Guide

Thank you for considering contributing to our project! We appreciate your efforts and look forward to collaborating with you. This guide outlines the steps you should follow to make the contribution process as smooth as possible.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Bugs and Requesting Features](#reporting-bugs-and-requesting-features)
- [Code Style and Linting](#code-style-and-linting)
- [Testing](#testing)
- [Documentation](#documentation)
- [Releasing](#releasing)

## Code of Conduct

By participating in this project, you agree to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please review the Code of Conduct before contributing to ensure a welcoming and inclusive environment for everyone.

## Getting Started

To start contributing to the project, follow these steps:

1. Fork the repository on GitHub.
2. Clone your fork to your local development environment.
3. Set up the development environment with [uv](https://docs.astral.sh/uv/): `uv sync --frozen` installs the package and the `dev` group (see [Dependency Workflow](#dependency-workflow-uv)) on Python 3.12, which `.python-version` selects to match CI and Read the Docs; `UV_PYTHON=3.10 uv sync` picks another version. To work on the Django app, follow [cs_app/README.md](cs_app/README.md).
4. Create a new branch for your changes, using a descriptive name.
5. Implement your changes, following the guidelines outlined in this document.
6. Ensure your changes pass all tests and comply with our code style guidelines.

### Dependency Workflow (uv)

- Manage dependencies in `pyproject.toml` only.
- Regenerate `uv.lock` after dependency changes using `uv lock`.
- Commit `pyproject.toml` and `uv.lock` in the same pull request.
- Development tools live in PEP 735 dependency groups, not in package extras.
  `uv sync` installs the `dev` group by default; add `--group web` for the Django app,
  `--group docs` for Sphinx and `--group e2e` for the Playwright tests, e.g.
  `uv sync --frozen --group web`. `uv sync` removes the groups you do not name,
  so combine them in one command (`uv sync --frozen --group docs --group web`).
- Before pushing, run the same checks as CI: `uv run ruff check .`,
  `uv run ruff format --check .`, `uv run mypy src/` and `uv run pytest -n auto`.

## Submitting a Pull Request

Once you have implemented your changes, you can submit a pull request. Please follow these steps:

1. Commit your changes in your feature branch.
2. Push your changes to your fork on GitHub.
3. Navigate to the original repository on GitHub, and click the "New Pull Request" button.
4. Select your fork and the appropriate branch.
5. Provide a detailed description of your changes, explaining what you have changed and why.
6. Ensure that the "Allow edits from maintainers" checkbox is checked.

We will review your pull request as soon as possible and provide feedback. If any changes are required, please make the necessary updates and push your changes to your branch. Once your pull request is approved, it will be merged into the main branch.

## Reporting Bugs and Requesting Features

If you encounter any bugs or issues, or have a feature request, please create an issue in the project's [issue tracker](https://github.com/BuczynskiRafal/catchments_simulation/issues). Be sure to provide as much information as possible to help us understand and address your issue or request. Report security problems privately as described in [SECURITY.md](SECURITY.md).

## Code Style and Linting

The code is formatted and linted with [ruff](https://docs.astral.sh/ruff/) (configured in `pyproject.toml`) and the package is type-checked with mypy in strict mode. Before pushing, run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/
```

`uv run pre-commit install` runs the same ruff and mypy checks on every commit. Docstrings use the NumPy style and document parameters, return values, units and raised exceptions.

## Testing

Tests are crucial for maintaining the quality and stability of the project. When submitting changes, please ensure that your changes pass all existing tests. If your changes introduce new functionality or modify existing functionality, please write new tests to cover these changes.

```bash
uv run pytest -n auto
DJANGO_SETTINGS_MODULE=cs_app.test_settings uv run --group web pytest -n auto cs_app/main/tests/ cs_app/register/tests.py
```

The tests run the real SWMM engine. On macOS, if Python is killed with SIGKILL when importing pyswmm, see the codesign note in [cs_app/README.md](cs_app/README.md#macos-processes-killed-with-sigkill).

## Documentation

Please update any relevant documentation as needed when making changes to the project. This includes comments in the code, as well as external documentation such as README files or user guides. Proper documentation ensures that users and fellow contributors can understand and effectively use the project.

User-visible changes to the package go into [CHANGELOG.md](CHANGELOG.md). The Sphinx documentation must build without warnings, as it does in CI:

```bash
uv sync --frozen --group docs --group web
uv run sphinx-build -W --keep-going -b html docs/source docs/build/html
```

## Releasing

Releases are published by the `Catchment Simulation` workflow (`.github/workflows/python-package.yml`) when a version tag is pushed.

1. Set the new version in `pyproject.toml` (`version = "X.Y.Z"`), following [Semantic Versioning](https://semver.org/).
2. In `CHANGELOG.md`, turn the unreleased section into `## [X.Y.Z] - YYYY-MM-DD` and update the link references at the bottom.
3. Merge to `main`, then tag that commit and push the tag: `git tag vX.Y.Z` and `git push origin vX.Y.Z`.
4. CI builds the sdist and the wheel, checks that the tag without its leading `v` equals the version in the built wheel (a mismatch fails the build), installs the wheel into a clean virtual environment as a smoke test, and publishes to PyPI with [Trusted Publishing](https://docs.pypi.org/trusted-publishers/). No API token is stored in the repository.

Pre-releases are rehearsals on TestPyPI: set a pre-release version such as `X.Y.Zrc1` in `pyproject.toml` and push the tag `vX.Y.Zrc1`. Any version that [packaging](https://packaging.pypa.io/) treats as a pre-release (`aN`, `bN`, `rcN`) is published to TestPyPI instead of PyPI.

One-time setup, needed before the first release with this workflow:

- On PyPI and on TestPyPI, add a trusted publisher for the project: owner `BuczynskiRafal`, repository `catchments_simulation`, workflow `python-package.yml`, and environment `pypi` (on PyPI) or `testpypi` (on TestPyPI).
- In the GitHub repository settings, create the environments `pypi` and `testpypi`. Requiring a reviewer on `pypi` adds a manual approval before every release.

Thank you for your interest in contributing to our project! We look forward to working with you and appreciate your help in making this project better for everyone.
