ifeq ($(OS), Windows_NT)
    RM = del /Q /F
    EXT = *.pyd
else
    RM = rm -rf
    EXT = *.so
endif

all: build

install:
    pip install -r requirements.txt

build:
    python setup.py build_ext --inplace

run: build
    python main.py

clean:
    $(RM) build
    $(RM) $(EXT)

.PHONY: all install build run clean