SHELL := /bin/bash

.PHONY: image install test-local test-docker clean pre-commit

image:
	docker build -t trace-classifier .

venv:
	virtualenv --python=python3.6 venv

install: venv
	source venv/bin/activate; \
	pip install -r requirements-dev.txt; \
	pip install -r requirements.txt

clean:
	rm -r venv/

test-local: install
	source venv/bin/activate; \
	pytest test/

test-docker: image
	docker run -i -t trace-classifier:latest

pre-commit: install
	source venv/bin/activate; \
	pre-commit run --all-files
