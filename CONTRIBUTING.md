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

## Code of Conduct

By participating in this project, you agree to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please review the Code of Conduct before contributing to ensure a welcoming and inclusive environment for everyone.

## Getting Started

To start contributing to the project, follow these steps:

1. Fork the repository on GitHub.
2. Clone your fork to your local development environment.
3. Set up the development environment following the instructions in the project README.
4. Create a new branch for your changes, using a descriptive name.
5. Implement your changes, following the guidelines outlined in this document.
6. Ensure your changes pass all tests and comply with our code style guidelines.

### Dependency Workflow (uv)

- Manage dependencies in `pyproject.toml` only.
- Regenerate `uv.lock` after dependency changes using `uv lock`.
- Commit `pyproject.toml` and `uv.lock` in the same pull request.
- Install local environments with `uv sync --frozen --extra dev --extra web`.

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

If you encounter any bugs or issues, or have a feature request, please create an issue in the project's [issue tracker](https://github.com/BuczynskiRafal/catchments_simulation/issues). Be sure to provide as much information as possible to help us understand and address your issue or request.

## Code Style and Linting

Please follow the code style and formatting guidelines provided in the project README. If the project uses a linter or code formatter, ensure that your changes pass any linting or formatting checks.

## Testing

Tests are crucial for maintaining the quality and stability of the project. When submitting changes, please ensure that your changes pass all existing tests. If your changes introduce new functionality or modify existing functionality, please write new tests to cover these changes.

## Documentation

Please update any relevant documentation as needed when making changes to the project. This includes comments in the code, as well as external documentation such as README files or user guides. Proper documentation ensures that users and fellow contributors can understand and effectively use the project.

Thank you for your interest in contributing to our project! We look forward to working with you and appreciate your help in making this project better for everyone.
