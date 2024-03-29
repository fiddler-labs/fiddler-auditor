PYTHONPATH := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

test:
	python -m spacy download en_core_web_trf
	python -m pytest -s -v  tests/

build:
	python3 -m build

lint:
	flake8 .

publish_test:
	python3 -m twine upload --non-interactive --repository-url https://test.pypi.org/legacy/ dist/*

publish:
	python3 -m twine upload --non-interactive dist/*

build_docs:
	cp examples/LLM_Evaluation.ipynb docs/source/
	cp examples/LLM_Evaluation_Azure.ipynb docs/source/
	cd docs/ && $(MAKE) clean html
clean:
	rm -rf dist/*
	rm -rf build/

all: test clean build build_docs
