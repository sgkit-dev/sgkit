# sgkit
Statistical genetics toolkit

## Developer documentation

### Build and test

From a Python virtual environment run:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

To check code coverage and get a coverage report, run

```bash
pytest --cov=sgkit
```

### Code standards

Use [pre-commit](https://pre-commit.com/) to check or enforce the coding standards. Install the git hook using:

```bash
pre-commit install
```

To manually enforce (or check) the source code adheres to our coding standards:

```bash
pre-commit run --all-files
mypy --strict .
```

To run specific tool (black/flake8/isort etc):

```bash
pre-commit run black --all-files
```

Notes:
 * pre-commit does not run the `mypy` check, so you will need to run that manually before submitting a pull request
 * if you skip `--all-files` checks are incremental 
