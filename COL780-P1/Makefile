# Makefile for AR Tag Project

# Detect OS for cleanup command
ifeq ($(OS),Windows_NT)
    RM = del /Q /F
    EXT = *.pyd
else
    RM = rm -rf
    EXT = *.so
endif

# Default target
all: build

# 1. Install Dependencies
install:
	pip install -r requirements.txt

# 2. Build the C++ Extension (In-place)
build:
	python setup.py build_ext --inplace

# 3. Run the Application
run: build
	python main.py

# 4. Clean build artifacts
clean:
	$(RM) build
	$(RM) $(EXT)

# Phony targets help avoid conflicts with file names
.PHONY: all install build run clean