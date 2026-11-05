PYTHON ?= .venv/bin/python
SF ?= code/situation_financiere.py
SF_ARGS ?=

.PHONY: plan pilot ocr build test

plan:
	$(PYTHON) $(SF) plan $(SF_ARGS)

pilot:
	$(PYTHON) $(SF) ocr --max-pages 3 $(SF_ARGS)

ocr:
	$(PYTHON) $(SF) ocr $(SF_ARGS)

build:
	$(PYTHON) $(SF) build $(SF_ARGS)

test:
	$(PYTHON) -m pytest -q tests/
