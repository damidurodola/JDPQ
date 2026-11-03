OS := $(shell uname -s)
ARCH := $(shell uname -m)
PKG_CONFIG ?= pkg-config

EIGEN_CFLAGS := $(shell $(PKG_CONFIG) --cflags eigen3 2>/dev/null)
GEOS_PC := $(shell $(PKG_CONFIG) --exists geos_c && echo geos_c || $(PKG_CONFIG) --exists geos && echo geos)
GEOS_CFLAGS_PKG := $(shell $(PKG_CONFIG) --cflags $(GEOS_PC) 2>/dev/null)
GEOS_LIBS_PKG := $(shell $(PKG_CONFIG) --libs $(GEOS_PC) 2>/dev/null)

ARCH_FLAG =
OPENMP_CFLAGS =
OPENMP_LIBS =
GEOS_CFLAGS =
GEOS_LIBS =
CXX_DEFAULT = clang++

ifeq ($(OS),Darwin)
	# Detect architecture for Homebrew path.
	ifeq ($(ARCH), arm64)
		BREW_PREFIX = /opt/homebrew
	else
		# On Apple Silicon machines running through Rosetta (`uname -m` reports
		# x86_64), GEOS is often only installed under /opt/homebrew. Detect that
		# situation and target arm64 explicitly so all libs match.
		HAS_ARM64_BREW := $(shell test -d /opt/homebrew/opt/geos && echo yes)
		ifeq ($(HAS_ARM64_BREW), yes)
			BREW_PREFIX = /opt/homebrew
			ARCH_FLAG = -arch arm64
		else
			BREW_PREFIX = /usr/local
		endif
	endif

	OPENMP_CFLAGS = -Xpreprocessor -fopenmp -I$(BREW_PREFIX)/opt/libomp/include
	OPENMP_LIBS = -L$(BREW_PREFIX)/opt/libomp/lib -lomp
	ifneq ($(strip $(GEOS_CFLAGS_PKG)),)
		GEOS_CFLAGS = $(GEOS_CFLAGS_PKG)
	else
		GEOS_CFLAGS = -I$(BREW_PREFIX)/opt/geos/include
	endif
	ifneq ($(strip $(GEOS_LIBS_PKG)),)
		GEOS_LIBS = $(GEOS_LIBS_PKG)
	else
		GEOS_LIBS = -L$(BREW_PREFIX)/opt/geos/lib -lgeos_c
	endif
else ifeq ($(OS),Linux)
	CXX_DEFAULT = g++
	OPENMP_CFLAGS = -fopenmp
	OPENMP_LIBS = -fopenmp
	GEOS_CFLAGS = $(GEOS_CFLAGS_PKG)
	ifneq ($(strip $(GEOS_LIBS_PKG)),)
		GEOS_LIBS = $(GEOS_LIBS_PKG)
	else
		GEOS_LIBS = -lgeos_c
	endif
else
	$(warning Unsupported OS '$(OS)'; attempting generic pkg-config build)
	OPENMP_CFLAGS = -fopenmp
	OPENMP_LIBS = -fopenmp
	GEOS_CFLAGS = $(GEOS_CFLAGS_PKG)
	GEOS_LIBS = $(GEOS_LIBS_PKG)
endif

CXX ?= $(CXX_DEFAULT)

# Tune for the build machine. The distance kernels (fvec_jaccard and friends)
# are the benchmark's hot path and are written to vectorize; without this the
# compiler targets baseline x86-64 and emits 128-bit SSE, leaving AVX2/AVX-512
# unused. Build with NATIVE=0 to produce a portable binary instead.
NATIVE ?= 1
NATIVE_FLAG =
ifeq ($(NATIVE),1)
	SUPPORTS_NATIVE := $(shell $(CXX) -march=native -E -x c++ /dev/null >/dev/null 2>&1 && echo yes)
	ifeq ($(SUPPORTS_NATIVE),yes)
		NATIVE_FLAG = -march=native
	endif
endif

# CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -fopenmp -I$(HOME)/local/include
# LDFLAGS = -fopenmp
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 $(ARCH_FLAG) $(NATIVE_FLAG) $(OPENMP_CFLAGS) $(EIGEN_CFLAGS) $(GEOS_CFLAGS)

LDFLAGS = $(ARCH_FLAG) $(OPENMP_LIBS) $(GEOS_LIBS)

# MPI support: build with  make USE_MPI=1
# mpicxx wraps the underlying compiler + MPI include/link flags automatically.
ifeq ($(USE_MPI),1)
  CXX = mpicxx
  CXXFLAGS += -DUSE_MPI
endif

SRC_DIR = src
BUILD_DIR = build
TARGET = test

FAISS_PREFIX ?= $(HOME)/faiss-cpp-env
SRC_DIR = src
BUILD_DIR = build
TOOLS = convert_to_binary extract_train export_train remove_train test_rerank test_rtree faiss_pq_ablation benchmark

.PHONY: all clean $(TOOLS)
all: convert_to_binary extract_train export_train remove_train test_rerank test_rtree benchmark

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

convert_to_binary: convert_to_binary.cpp
	$(CXX) -std=c++17 -Wall -Wextra -O3 $(ARCH_FLAG) $(NATIVE_FLAG) -o $@ $<
extract_train: $(BUILD_DIR)/readData.o extract_train.cpp src/train_ids.h src/db_compact.h
	$(CXX) $(CXXFLAGS) -o $@ extract_train.cpp $(BUILD_DIR)/readData.o $(LDFLAGS)
export_train: $(BUILD_DIR)/readData.o export_train.cpp src/train_ids.h
	$(CXX) $(CXXFLAGS) -o $@ export_train.cpp $(BUILD_DIR)/readData.o $(LDFLAGS)
remove_train: $(BUILD_DIR)/readData.o remove_train.cpp src/train_ids.h src/db_compact.h
	$(CXX) $(CXXFLAGS) -o $@ remove_train.cpp $(BUILD_DIR)/readData.o $(LDFLAGS)
test_rerank: $(BUILD_DIR)/PolygonGT.o $(BUILD_DIR)/readData.o $(BUILD_DIR)/index.o $(BUILD_DIR)/distances.o test_rerank.cpp
	$(CXX) $(CXXFLAGS) -o $@ test_rerank.cpp $(BUILD_DIR)/PolygonGT.o $(BUILD_DIR)/readData.o $(BUILD_DIR)/index.o $(BUILD_DIR)/distances.o $(LDFLAGS)
test_rtree: $(BUILD_DIR)/RTreeSearch.o $(BUILD_DIR)/PolygonGT.o $(BUILD_DIR)/index.o test_rtree.cpp
	$(CXX) $(CXXFLAGS) -o $@ test_rtree.cpp $(BUILD_DIR)/RTreeSearch.o $(BUILD_DIR)/PolygonGT.o $(BUILD_DIR)/index.o $(LDFLAGS)
# Method-agnostic benchmark driver. Without the JD-PQ plugin only the exact "flat"
# reference method is available (ANN_METHOD_A=flat ANN_METHOD_B=); to add JD-PQ,
# replace jdpq_stub.o by the released jdpq_index.o and its core objects.
BENCH_OBJS = $(BUILD_DIR)/benchmark.o $(BUILD_DIR)/ann_registry.o $(BUILD_DIR)/jdpq_stub.o \
             $(BUILD_DIR)/indexFlat.o $(BUILD_DIR)/index.o $(BUILD_DIR)/distances.o $(BUILD_DIR)/readData.o
benchmark: $(BENCH_OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)
# FAISS (Euclidean) PQ/IVFPQ baseline; needs libfaiss under FAISS_PREFIX.
faiss_pq_ablation: $(BUILD_DIR)/readData.o $(BUILD_DIR)/distances.o faiss_pq_ablation.cpp
	$(CXX) $(CXXFLAGS) -I$(FAISS_PREFIX)/include -o $@ faiss_pq_ablation.cpp $(BUILD_DIR)/readData.o $(BUILD_DIR)/distances.o \
		-L$(FAISS_PREFIX)/lib -Wl,-rpath,$(FAISS_PREFIX)/lib -lfaiss $(LDFLAGS)
clean:
	rm -rf $(BUILD_DIR) $(TOOLS)
