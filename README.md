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
```

To run specific tool (`black`/`flake8`/`isort`/`mypy` etc):

```bash
pre-commit run black --all-files
```

Notes:
 * if you skip `--all-files` checks are incremental
