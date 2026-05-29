PYTHON ?= .venv/bin/python
EXPORTS = OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE

.PHONY: data
data:
	$(EXPORTS) $(PYTHON) -m phonon_omegamax.cli data

.PHONY: train-gbdt
train-gbdt:
	$(EXPORTS) $(PYTHON) -m phonon_omegamax.cli train-gbdt

.PHONY: train-cgcnn
train-cgcnn:
	$(EXPORTS) $(PYTHON) -m phonon_omegamax.cli train-cgcnn

.PHONY: figures
figures:
	$(EXPORTS) $(PYTHON) scripts/make_figures.py

.PHONY: all
all: data train-gbdt train-cgcnn figures

.PHONY: test
test:
	$(EXPORTS) $(PYTHON) -m pytest -q
