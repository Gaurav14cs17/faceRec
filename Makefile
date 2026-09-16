# Usage: make install | doctor | test | verify
PYTHON ?= python
SAMPLE ?= data/samples/t1.jpg

.PHONY: install doctor weights test verify index-verify help

help:
	@echo "Targets: install, weights, doctor, test, verify, index-verify"

install:
	$(PYTHON) -m pip install -r requirements.txt

weights:
	$(PYTHON) run.py weights

doctor:
	$(PYTHON) run.py doctor

test:
	$(PYTHON) run.py test

verify:
	$(PYTHON) run.py verify --image $(SAMPLE)

index-verify:
	$(PYTHON) run.py index-verify --image $(SAMPLE)
