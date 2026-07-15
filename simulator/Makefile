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

# GEMM problem dims for the timeloop mapping search — override on the command
# line, e.g.: make wl_sim PROB_M=2048 PROB_K=512
PROB_M ?= 1024
PROB_K ?= 1024
PROB_N ?= 1024

WORKLOAD_BIN ?= workload.bin
MAP_ARGS     ?=

.PHONY: all gemm wl sweep wl_sim clean

all: gemm_sim.exe

gemm_sim.exe: gemm_cycle_accurate_sim.cpp
	$(CXX) $(CXXFLAGS) $(DEFINES) $< -o $@

# Shorthand alias (rebuilds binary if source changed, then runs workload)
gemm: gemm_sim.exe

wl: gemm_sim.exe
	uv run python3 llm_workload.py $(WL_ARGS)

sweep:
	uv run sweep.py

# Find the optimal timeloop mapping for this array config, then feed it
# straight into the cycle-accurate sim as a --binary-file workload.
wl_sim: gemm_sim.exe
	uv run python3 find_mapping.py \
		--M $(PROB_M) --K $(PROB_K) --N $(PROB_N) \
		--p-rows $(CFG_P_ROWS) --p-cols $(CFG_P_COLS) \
		--n-h $(CFG_N_H) --n-w $(CFG_N_W) \
		--k-depth $(CFG_K) --m-parallel $(CFG_M) --mag-bits $(CFG_MAG_BITS) \
		--workload-out $(WORKLOAD_BIN) $(MAP_ARGS)
	./gemm_sim.exe --binary-file $(WORKLOAD_BIN)

clean:
	rm -f *.exe
	rm -f *.out
	rm -f *.csv
	rm -f $(WORKLOAD_BIN)
