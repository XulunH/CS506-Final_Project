# Variables
PYTHON := python3.11
VENV := .venv
SCRIPT := all_in_one.py
REQUIREMENTS := requirements.txt
DATASET := revised_traffic_dataset.csv
OUTPUT_DIR := ./outputs

# Rules
all: setup_venv install run

# Set up virtual environment
setup_venv:
	@echo "Creating virtual environment in $(VENV)..."
	$(PYTHON) -m venv $(VENV)

# Install dependencies in virtual environment
install: setup_venv
	@echo "Activating virtual environment and installing dependencies..."
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r $(REQUIREMENTS)

# Run the script within virtual environment
run: 
	@echo "Running the script in virtual environment..."
	$(VENV)/bin/$(PYTHON) $(SCRIPT)

# Clean up output files
clean:
	@echo "Cleaning up output files..."
	rm -rf $(OUTPUT_DIR)/*.png

# Ensure output directory exists
setup:
	@echo "Creating output directory..."
	mkdir -p $(OUTPUT_DIR)

# Run the script in test mode (within virtual environment)
test:
	@echo "Running script in test mode (without outputs)..."
	$(VENV)/bin/$(PYTHON) $(SCRIPT) --test

.PHONY: all setup_venv install run clean setup test