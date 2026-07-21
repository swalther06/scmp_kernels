# Recipes use bash features (source, [[ ]]) and the EDA target sources a profile.
SHELL := /bin/bash

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
# line, e.g.: make mapping PROB_M=2048 PROB_K=512
PROB_M ?= 1024
PROB_K ?= 1024
PROB_N ?= 1024

WORKLOAD_BIN ?= workload.bin
MAP_ARGS     ?=

# --- Timeloop+Accelergy energy-optimal mapping (container flow) -------------
# Self-contained: needs only the docker daemon + the cached timeloop-accelergy
# image. `make configs` generates the v4 YAML; `make mapping` runs the mapper on
# them with the measured SC-PE energy from the plug-in. Results in
# timeloop_container_run/out/.
PLUGIN_DIR := accelergy-sc-plugin
TL_IMG     := timeloopaccelergy/timeloop-accelergy-pytorch:latest-amd64
TL_DIR     := timeloop_container_run
CFG_DIR    := timeloop_workspace
# GLB=1 (default): 3-level arch with on-chip buffer (reuse-optimal mapping).
# GLB=0: drop the buffer -> simple-tiling baseline (operands streamed from DRAM).
GLB   ?= 1
NOGLB := $(if $(filter 0,$(GLB)),--no-glb,)

# --- Inner-PE energy characterization (ASTRAEA RTL flow) --------------------
# Cross-repo, needs the ASTRAEA EDA environment (VCS + Design Compiler +
# NanGate45 PDK; modules from ASTRAEA/env_setup.sh). Produces the pJ/MAC that
# feeds sc_mac.py's E_COMPUTE_J.
ASTRAEA_DIR  ?= ../../ASTRAEA
INNER_TARGET ?= NanGate45/INNER_PE_K32M8N4
INNER_TB     ?= designs/sc_gemm_3d/power_pe_single_int_2bit_lane_K32M8N4.sv
INNER_SAIF   := build/$(INNER_TB)/dut.saif

.PHONY: all gemm wl sweep wl_sim configs \
        mapping mapping_baseline mapping_compare inner_pe_energy \
        help clean

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
# (Native-mapper path via find_mapping.py; needs a built timeloop-mapper binary.
#  For the container-based energy-optimal search instead, use `make mapping`.)
wl_sim: gemm_sim.exe
	uv run python3 find_mapping.py \
		--M $(PROB_M) --K $(PROB_K) --N $(PROB_N) \
		--p-rows $(CFG_P_ROWS) --p-cols $(CFG_P_COLS) \
		--n-h $(CFG_N_H) --n-w $(CFG_N_W) \
		--k-depth $(CFG_K) --m-parallel $(CFG_M) --mag-bits $(CFG_MAG_BITS) \
		--workload-out $(WORKLOAD_BIN) $(MAP_ARGS)
	./gemm_sim.exe --binary-file $(WORKLOAD_BIN)

# ---------------------------------------------------------------------------
# Accelergy plug-in + Timeloop energy-optimal mapping
# ---------------------------------------------------------------------------

# STEP 1: generate the 4 timeloopfe-v4 config files into $(CFG_DIR).
# Override problem/array: make configs PROB_M=512 CFG_P_ROWS=2 ... ; GLB=0 for baseline.
configs:
	uv run python3 find_mapping.py --emit-configs --work-dir $(CFG_DIR) \
		--M $(PROB_M) --N $(PROB_N) --K $(PROB_K) \
		--p-rows $(CFG_P_ROWS) --p-cols $(CFG_P_COLS) \
		--n-h $(CFG_N_H) --n-w $(CFG_N_W) \
		--k-depth $(CFG_K) --m-parallel $(CFG_M) --mag-bits $(CFG_MAG_BITS) $(NOGLB)

# STEP 2: run Timeloop+Accelergy (container) on the generated configs.
# Energy-optimal mapping with the measured SC-PE energy. Results in $(TL_DIR)/out/.
#   make mapping                    # 1024^3 (or PROB_*), reuse-optimal
#   make mapping PROB_M=256 PROB_N=256 PROB_K=256
#   make mapping GLB=0              # simple-tiling baseline
mapping: configs
	@rm -rf $(TL_DIR)/work && mkdir -p $(TL_DIR)/work/out $(TL_DIR)/work/configs
	@cp $(CFG_DIR)/arch.yaml $(CFG_DIR)/problem.yaml $(CFG_DIR)/mapper.yaml \
	    $(CFG_DIR)/constraints.yaml $(TL_DIR)/work/configs/
	@cp -r $(PLUGIN_DIR) $(TL_DIR)/work/plugin
	@cp $(TL_DIR)/run.py $(TL_DIR)/work/run.py
	@chmod -R 777 $(TL_DIR)/work
	docker run --rm -v $(CURDIR)/$(TL_DIR)/work:/work $(TL_IMG) \
		bash -lc 'python3 /work/run.py /work/configs /work/out' 2>&1 \
		| grep -avE 's6-|cont-init|services\.d|fix-attrs|adduser|usermod|User (uid|gid)|Jupter|jupyter|token|lab\?|Serving|open this file|127\.0\.0\.1|^-+$$|^ *$$' || true
	@rm -rf $(TL_DIR)/out && cp -r $(TL_DIR)/work/out $(TL_DIR)/out
	@echo "=== mapping summary (GLB=$(GLB)) ==="
	@grep -iE "Cycles:|Energy:.*uJ|Utilization:" $(TL_DIR)/out/timeloop-mapper.stats.txt 2>/dev/null \
		|| echo "no stats -- see $(TL_DIR)/out/err.txt"

# Simple-tiling baseline (no on-chip buffer). Compare energy to `make mapping`.
mapping_baseline:
	@$(MAKE) mapping GLB=0

# Run both and print the total energy each achieved (optimal vs simple tiling).
mapping_compare:
	@$(MAKE) mapping GLB=1
	@cp $(TL_DIR)/out/timeloop-mapper.stats.txt /tmp/tl_opt.stats.txt
	@$(MAKE) mapping GLB=0
	@echo "===================== optimal vs simple tiling ====================="
	@printf "optimal (with buffer) : "; grep -iE "Energy:.*uJ" /tmp/tl_opt.stats.txt | tail -1
	@printf "simple tiling (no buf): "; grep -iE "Energy:.*uJ" $(TL_DIR)/out/timeloop-mapper.stats.txt | tail -1

# ---------------------------------------------------------------------------
# Inner-PE energy characterization (ASTRAEA RTL: synth -> GL sim + SAIF -> power)
# ---------------------------------------------------------------------------
# Requires the ASTRAEA EDA env. If `module` is not available in this recipe's
# shell, load the Synopsys modules first (or run these three steps by hand in
# $(ASTRAEA_DIR)). Prints total power; energy/MAC = P_avg * t_window / N_MACs
# (see accelergy-sc-plugin/sc_mac.py for the divisor).
inner_pe_energy:
	cd $(ASTRAEA_DIR) && source ./env_setup.sh && \
		$(MAKE) synth TARGET=$(INNER_TARGET) && \
		$(MAKE) sim   TB=$(INNER_TB) GL=syn TARGET=$(INNER_TARGET) && \
		$(MAKE) power TARGET=$(INNER_TARGET) SAIF=$(INNER_SAIF)

help:
	@echo "Targets:"
	@echo "  all / gemm       build gemm_sim.exe (cycle-accurate simulator)"
	@echo "  wl               run the LLM workload through the simulator"
	@echo "  sweep            run the DSE sweep"
	@echo "  wl_sim           find_mapping.py (native mapper) -> binary workload -> sim"
	@echo "  configs          generate the 4 timeloopfe-v4 YAML files -> $(CFG_DIR)/"
	@echo "  mapping          energy-optimal GEMM mapping (configs + Timeloop, container)"
	@echo "  mapping_baseline same, but simple-tiling baseline (GLB=0, no buffer)"
	@echo "  mapping_compare  run both and print the energy comparison"
	@echo "  inner_pe_energy  characterize inner-PE pJ/MAC via ASTRAEA RTL (needs EDA env)"
	@echo "  clean            remove build artifacts + mapping outputs"
	@echo ""
	@echo "Common overrides: PROB_M/PROB_N/PROB_K (problem), CFG_* (array config),"
	@echo "                  GLB=0 (drop on-chip buffer), ASTRAEA_DIR/INNER_TARGET (chars.)."

clean:
	rm -f *.exe
	rm -f *.out
	rm -f *.csv
	rm -f $(WORKLOAD_BIN)
	rm -rf timeloop_container_run/out timeloop_container_run/work
	rm -f $(PLUGIN_DIR)/*.pyc
	rm -rf $(PLUGIN_DIR)/__pycache__ $(PLUGIN_DIR)/build $(PLUGIN_DIR)/*.egg-info
