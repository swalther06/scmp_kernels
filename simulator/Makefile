CXX      := g++
CXXFLAGS := -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra

# Architecture knobs — override on the command line, e.g.:
#   make CFG_K=8 CFG_P_ROWS=2 CFG_M=4
CFG_MAG_BITS ?= 7
CFG_K        ?= 4
CFG_P_ROWS   ?= 4
CFG_P_COLS   ?= 4
CFG_N_H      ?= 1
CFG_N_W      ?= 1
CFG_M        ?= 1
CFG_OWEN     ?= 0

DEFINES := \
	-DCFG_MAG_BITS=$(CFG_MAG_BITS) \
	-DCFG_K=$(CFG_K) \
	-DCFG_P_ROWS=$(CFG_P_ROWS) \
	-DCFG_P_COLS=$(CFG_P_COLS) \
	-DCFG_N_H=$(CFG_N_H) \
	-DCFG_N_W=$(CFG_N_W) \
	-DCFG_M=$(CFG_M) \
	-DCFG_OWEN=$(CFG_OWEN)

# Workload args — append on the command line, e.g.:
#   make wl WL_ARGS="--granularity per_row --output results.csv"
WL_ARGS ?=

.PHONY: all gemm wl sweep clean

all: gemm_sim.exe

gemm_sim.exe: gemm_cycle_accurate_sim.cpp
	$(CXX) $(CXXFLAGS) $(DEFINES) $< -o $@

# Shorthand alias (rebuilds binary if source changed, then runs workload)
gemm: gemm_sim.exe

wl: gemm_sim.exe
	uv run python3 llm_workload.py $(WL_ARGS)

sweep:
	uv run sweep.py

clean:
	rm -f *.exe
	rm -f *.out
	rm -f *.csv
