SHELL := bash
PACKAGE_NAME = $(shell basename "$$PWD")
PYTHON_NAME = $(shell echo "$(PACKAGE_NAME)" | sed -e 's/-//' | sed -e 's/-/_/g')
SOURCE = $(PYTHON_NAME)
PYTHON_FILES = $(SOURCE)/*.py
SHELL_FILES = bin/* debian/bin/* *.sh
PIP_INSTALL ?= install
DOWNLOAD_DIR = download

.PHONY: reformat check dist venv downloads

all: venv

# -----------------------------------------------------------------------------
# Python
# -----------------------------------------------------------------------------

reformat:
	scripts/format-code.sh $(PYTHON_FILES)

check:
	scripts/check-code.sh $(PYTHON_FILES)

venv: downloads
	scripts/create-venv.sh

dist: sdist

sdist:
	python3 setup.py sdist

test:
	scripts/run-tests.sh

# -----------------------------------------------------------------------------
# Downloads
# -----------------------------------------------------------------------------

# Rhasspy development dependencies
RHASSPY_DEPS := $(shell grep '^rhasspy-' requirements.txt | sort | comm -3 - rhasspy_wheels.txt | sed -e 's|^|$(DOWNLOAD_DIR)/|' -e 's/==/-/' -e 's/$$/.tar.gz/')

$(DOWNLOAD_DIR)/%.tar.gz:
	mkdir -p "$(DOWNLOAD_DIR)"
	scripts/download-dep.sh "$@"

downloads: $(RHASSPY_DEPS)
