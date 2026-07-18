# timeloop_container_run

Runs Timeloop + Accelergy inside the vendored container to find the
energy-optimal GEMM mapping, using the measured SC-PE energy from
[`../accelergy-sc-plugin`](../accelergy-sc-plugin). **Driven from the Makefile
one directory up — there is no shell script here.**

```bash
cd ..                            # scmp_kernels/simulator
make configs                     # 1. generate the 4 v4 YAML files -> timeloop_workspace/
make mapping                     # 2. run the mapper on them (container) -> optimal mapping
make mapping GLB=0               #    simple-tiling baseline (no on-chip buffer)
make mapping PROB_M=256 PROB_N=256 PROB_K=256
```

Results land in `./out/` (`timeloop-mapper.stats.txt`, `.map.txt`, `.ERT.yaml`,
…). The `MACC` compute energy in the ERT is the plug-in's measured pJ/MAC.

## Pieces

- **`run.py`** — the in-container driver (invoked by `make mapping`). All Python,
  no shell: it copies the plug-in to `/tmp` and `pip install`s it (so pip's
  build artifacts don't land on the host mount), patches timeloopfe's
  `COMPUTE_CLASSES` to register `sc_mac_inner` as arithmetic, then loads the v4
  configs and calls the mapper.
- **`../find_mapping.py --emit-configs`** — generates the 4 v4 files
  (`make configs`), with a divisibility check (M/N/K must fit the array).

## Why a container

Neither Timeloop nor Accelergy is installed on the host and they can't be built
here (no scons/boost/yaml-cpp; no PyPI for accelergy). The cached image
`timeloopaccelergy/timeloop-accelergy-pytorch:latest-amd64` bundles the mapper,
Accelergy, CACTI, and pytimeloop. Only the Docker daemon is needed.

## v4 schema gotchas (baked into find_mapping.py, noted here for reference)

- `variables` values are **expression-evaluated** — only put numbers there;
  `technology` ("45nm") lives as a **quoted** attribute on the top container
  (`QuotedStr`), and must be on a node all components inherit (Accelergy
  requires it on every component, incl. DRAM).
- Storage nodes need one of `depth`/`entries`/`sizeKB`.
- A custom compute class (`sc_mac_inner`) parses as Storage unless registered in
  `COMPUTE_CLASSES` (run.py patches this).
- Spatial constraints need `split` + `permutation` to assign factors to the
  meshX/meshY axes; a temporal constraint on the spatial (`PE`) level is
  unsatisfiable (its temporal factors are forced to 1).

## Scope

The `rng_bank`/SNG outer level is omitted (timeloopfe rejects a custom *storage*
class); the outer 1.5x term is mapping-invariant and added post-hoc. See the
plug-in README for the inner/outer split.
