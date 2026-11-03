# Makefile for Entropic (fallback when CMake unavailable)
CC = gcc
CXX = g++
CFLAGS = -std=c11 -O3 -Wall -Wextra -Iinclude
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -Iinclude
LDFLAGS = -lm

# Optional FFTW
HAVE_FFTW := $(shell pkg-config --exists fftw3 && echo 1 || echo 0)
ifeq ($(HAVE_FFTW),1)
CFLAGS += -DWFC_USE_FFTW $(shell pkg-config --cflags fftw3)
CXXFLAGS += -DWFC_USE_FFTW $(shell pkg-config --cflags fftw3)
LDFLAGS += $(shell pkg-config --libs fftw3)
endif

# Default: all features enabled
CFLAGS += -DWFC_ENABLE_AVX2 -DWFC_ENABLE_COPULA -DWFC_ENABLE_BACKTRACK
CXXFLAGS += -DWFC_ENABLE_AVX2 -DWFC_ENABLE_COPULA -DWFC_ENABLE_BACKTRACK

BINS = test_wfc bench_wfc example_basic example_3d example_cpp
ifeq ($(HAVE_FFTW),1)
BINS += example_spectral example_copula
endif

.PHONY: all clean test check

all: $(BINS)

test_wfc: tests/test_wfc.c include/wfc.h
	$(CC) $(CFLAGS) -o $@ tests/test_wfc.c $(LDFLAGS)

bench_wfc: benchmarks/bench_wfc.c include/wfc.h
	$(CC) $(CFLAGS) -o $@ benchmarks/bench_wfc.c $(LDFLAGS)

example_basic: examples/example_basic.c include/wfc.h
	$(CC) $(CFLAGS) -o $@ examples/example_basic.c $(LDFLAGS)

example_3d: examples/example_3d.c include/wfc.h
	$(CC) $(CFLAGS) -o $@ examples/example_3d.c $(LDFLAGS)

example_cpp: examples/example_cpp.cpp include/wfcpp.hpp include/wfc.h
	$(CXX) $(CXXFLAGS) -o $@ examples/example_cpp.cpp $(LDFLAGS)

ifeq ($(HAVE_FFTW),1)
example_spectral: examples/example_spectral.c include/wfc.h
	$(CC) $(CFLAGS) -o $@ examples/example_spectral.c $(LDFLAGS)

example_copula: examples/example_copula.c include/wfc.h
	$(CC) $(CFLAGS) -o $@ examples/example_copula.c $(LDFLAGS)
endif

test: test_wfc
	./test_wfc

check: test

clean:
	rm -f $(BINS)
