# Upgrading Fiddler

## Poetry upgrade

These instructions create a version of the **pyproject.toml** file to be compatible with the poetry build system instead of setuptools

```
[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "auditor"
version = "0.0.5"
authors = ["Fiddler Labs <support@fiddler.ai>"]
description = "Auditing large language models made easy."
readme = "README.md"
license = "Elastic License 2.0 (ELv2)"
repository = "https://github.com/fiddler-labs/fiddler-auditor"

[tool.poetry.dependencies]
python = ">=3.8.1<4.0"
notebook = "^6.0.1"
fiddler-checklist = "0.0.1"
pandas = "^1.3.5"
spacy-transformers = "^1.1.8"
jinja2 = "3.1.2"
langchain = ">=0.0.158,<=0.0.330"
openai = ">=0.27.0,<=0.28.1"
sentence-transformers = "^2.2.2"
tqdm = "^4.66.1"
httplib2 = "~0.22.0"

[tool.poetry.dev-dependencies]
pytest = "*"
build = "*"
twine = "*"
flake8 = "*"

```

## Install Poetry dependencies

This will set up the virtual environment, install Python and begin to install dependencies

    poetry install

### Install iso-639

Poetry will complain about a compatibility issue with fiddler-checklist and iso-639.  To get around this, install iso-639 first using pip

    poetry run pip install iso-639 

### Install poetry dependencies

    poetry install --with=dev

### Upgrade packages

    poetry add openai@latest
    poetry add langchain@latest
    poetry add spacy-transformers@latest
    poetry add sentence-transformers@latest

### Add Langchain Community

This adds support for openai and other models

    poetry add langchain-openai@latest

## Build auditor package

    poetry build

This should create the /dist directory and create
the wheel and tar.gz package assets

## Run tests

    poetry run pytest -v -s

This should set up the test environment and run all the tests in the /tests folder