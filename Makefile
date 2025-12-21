.PHONY: setup install train classify clean help

PYTHON_VERSION := 3.11.7
VENV_NAME := email-filter-env
PYTHON := $(VENV_NAME)/bin/python
PIP := $(VENV_NAME)/bin/pip

help:
	@echo "Unwanted Email Filter - Makefile Commands"
	@echo ""
	@echo "  make setup      - Install Python version and create virtual environment"
	@echo "  make install    - Install dependencies"
	@echo "  make train      - Train the DistilBERT email filter model"
	@echo "  make classify   - Run DistilBERT classification"
	@echo "  make llm        - Run LLM classifier on a file (usage: make llm FILE=path/to/email)"
	@echo "  make test       - Run pytest tests"
	@echo "  make clean      - Remove virtual environment and cached files"
	@echo "  make clean-all  - Remove everything including trained model"

setup:
	@echo "=== Setting up Email Filter ==="
	@if ! command -v pyenv &> /dev/null; then \
		echo "❌ pyenv is not installed. Install it with:"; \
		echo "   curl https://pyenv.run | bash"; \
		exit 1; \
	fi
	@echo "✓ pyenv found"
	@if ! pyenv versions | grep -q "$(PYTHON_VERSION)"; then \
		echo "📦 Installing Python $(PYTHON_VERSION)..."; \
		pyenv install $(PYTHON_VERSION); \
	else \
		echo "✓ Python $(PYTHON_VERSION) already installed"; \
	fi
	@echo "🔧 Setting local Python version..."
	@pyenv local $(PYTHON_VERSION)
	@echo "🌐 Creating virtual environment..."
	@python -m venv $(VENV_NAME)
	@echo "✅ Setup complete! Run 'make install' to install dependencies"

install: $(VENV_NAME)/bin/activate
	@echo "📦 Installing dependencies..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed!"

$(VENV_NAME)/bin/activate:
	@echo "⚠️  Virtual environment not found. Run 'make setup' first."
	@exit 1

train-distilbert: $(VENV_NAME)/bin/activate
	@echo "🚀 Training email filter model..."
	@$(PYTHON) email_train_distilbert.py

classify-distilbert: $(VENV_NAME)/bin/activate
	@echo "📧 Classifying emails..."
	@$(PYTHON) email_classify_distilbert.py

llm: $(VENV_NAME)/bin/activate
	@echo "🤖 Running LLM classifier..."
	@if [ -z "$(FILE)" ]; then \
		$(PYTHON) email_classify_llm.py; \
	else \
		$(PYTHON) email_classify_llm.py $(FILE); \
	fi

test: $(VENV_NAME)/bin/activate
	@echo "🧪 Running tests..."
	@$(PYTHON) -m pytest test_classifiers.py -v

clean:
	@echo "🧹 Cleaning up..."
	@rm -rf $(VENV_NAME)
	@rm -rf __pycache__
	@rm -rf .pytest_cache
	@rm -f .python-version
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@echo "✅ Cleaned virtual environment and cache files"

clean-all: clean
	@echo "🗑️  Removing trained model..."
	@rm -rf email_filter_model
	@echo "✅ Removed everything including trained model"