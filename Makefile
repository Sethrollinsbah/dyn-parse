# Makefile for the AI-Powered Dynamic Parser

# Compiler and flags
CARGO := cargo

# Default target
.PHONY: all
all: build

# Build the project in release mode for production
.PHONY: build
build:
	@echo "Building the project in release mode..."
	$(CARGO) build --release

# Run the tests, including ignored tests
.PHONY: test
test:
	@echo "Running tests..."
	$(CARGO) test --release

# Check the code for errors without building
.PHONY: check
check:
	@echo "Checking the code for errors..."
	$(CARGO) check

# Format the code according to Rust standards
.PHONY: fmt
fmt:
	@echo "Formatting the code..."
	$(CARGO) fmt

# Lint the code with Clippy for common mistakes
.PHONY: clippy
clippy:
	@echo "Linting the code with Clippy..."
	$(CARGO) clippy

# Clean up build artifacts
.PHONY: clean
clean:
	@echo "Cleaning up build artifacts..."
	$(CARGO) clean

# Help command to display available targets
.PHONY: help
help:
	@printf "Available commands:\n  make build    - Build the project in release mode\n  make run      - Run the application (see note in Makefile)\n  make test     - Run the test suite\n  make check    - Check the code without compiling\n  make fmt      - Format the source code\n  make clippy   - Lint the code with Clippy\n  make clean    - Remove build artifacts\n  make help     - Show this help message\n"

