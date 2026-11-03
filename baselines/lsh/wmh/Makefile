# Build: make            (binaries in wmh/bin)
#        make test       (unit tests)
CXX      = g++-14
UPSTREAM := ../third_party/dartminhash
XXHASH   := ../third_party/xxhash
# -isystem ../third_party lets upstream's #include "xxhash/xxhash.h" resolve to
# third_party/xxhash without placing files inside the upstream checkout.
CXXFLAGS := -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra -Iinclude -isystem $(UPSTREAM) -isystem ../third_party \
            -DWMH_WITH_BAGMINHASH
LDLIBS   := $(XXHASH)/libxxhash.a
PYTHON   ?= python3

HDRS := $(wildcard include/*.hpp) Makefile
BIN  := bin

all: $(BIN)/sanity $(BIN)/groundtruth $(BIN)/search $(BIN)/test_core

$(BIN)/sketch.o: src/sketch.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN)/sanity: src/sanity.cpp $(BIN)/sketch.o $(HDRS)
	$(CXX) $(CXXFLAGS) $< $(BIN)/sketch.o $(LDLIBS) -o $@

$(BIN)/groundtruth: src/groundtruth.cpp $(HDRS) | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN)/search: src/search.cpp $(BIN)/sketch.o $(HDRS)
	$(CXX) $(CXXFLAGS) $< $(BIN)/sketch.o $(LDLIBS) -o $@

$(BIN)/test_core: tests/test_core.cpp $(BIN)/sketch.o $(HDRS)
	$(CXX) $(CXXFLAGS) $< $(BIN)/sketch.o $(LDLIBS) -o $@

tests/fixture.npz: tests/make_fixture.py
	$(PYTHON) -P $< $@

test: $(BIN)/test_core tests/fixture.npz
	./$(BIN)/test_core tests/fixture.npz

$(BIN):
	mkdir -p $@

clean:
	rm -rf $(BIN) tests/fixture.npz

.PHONY: all test clean
