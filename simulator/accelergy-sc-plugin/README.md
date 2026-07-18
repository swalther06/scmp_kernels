# accelergy-sc-plugin

An Accelergy estimator plug-in for the **unary-stochastic-computing (SC) MAC**
used by `gemm_cycle_accurate_sim.cpp` / `find_mapping.py`: a bit-serial AND-gate
multiply feeding a shared accumulator, with stochastic-number generators (SNG)
converting operands to length-T bitstreams.

> **Status: skeleton.** The plug-in structure and Accelergy interface are real
> and runnable, but the hardware coefficients in `sc_mac.py` (`E_*_BIT`, leakage,
> area) are **placeholders**. Replace them with characterized numbers (RTL
> synthesis / SPICE / SCArch `power_array_3d.sv`) before trusting absolute
> energies. Until then it is useful for *relative* mapping comparison only.

## Why a plug-in (vs hand-writing ERT tables)

Accelergy still models the *entire non-novel* memory/interconnect hierarchy
(SRAM global buffer, register file, DRAM) with its built-in CACTI/library
estimators — often the dominant share of accelerator energy. This plug-in adds
the *one* novel component: the SC PE. Encoding it as a parameterized estimator
(vs a static ERT) means Accelergy re-evaluates it automatically as
`find_mapping.py`'s DSE sweeps `mag_bits`, width, tech node, etc.

## The one SC-specific modelling rule

Timeloop counts **logical** MACs — one `compute` action per scalar
multiply-accumulate, with no notion of a bitstream. So the length-T scaling of
an SC MAC lives inside the per-`compute` *energy* here, **not** in the action
count:

```
energy(compute) ≈ T · (AND-gate + accumulate + SNG) per-bit cost
```

This is the exact analogue of what `find_mapping.apply_hardware_timing()` does
for *cycles* (scaling by `cycles_per_mac_window`). Keep the T-factor in the ERT
number and the mapper stays correct.

## Actions exposed

| action         | meaning                                                        |
|----------------|----------------------------------------------------------------|
| `compute`      | one full SC MAC (length-T stream multiply + accumulate)        |
| `gated_compute`| operand zero/invalid, datapath gated: SNG/enable overhead only |
| `leak`         | static leakage per global cycle                                |

`compute`/`leak` match the v0.4 `intmac` action vocabulary Timeloop expects.
(The `mac_random`/`mac_reused`/`mac_gated` names you may have seen belong to the
older 2020-ispass *primitive* `intmac`; the reuse distinction here is the
`n_sng_per_mac` attribute instead — 2.0 = both operands fresh, 1.0 = one stream
reused, <1.0 = SNG shared across a row.)

## Install

```bash
cd accelergy-sc-plugin
pip3 install .            # installs into share/accelergy/estimation_plug_ins/
```

For a quick local test without pip, from a Python shell in this dir:
`from sc_mac import SCMac; SCMac.quick_install_this_file()`.

Verify Accelergy sees it:
```bash
accelergy --list-plug-ins    # sc_mac should appear
```

## Wire into the architecture

In the arch that `find_mapping.build_arch_spec()` emits, change the `MACC`
component from the built-in `intmac` to `sc_mac` and pass the SC knobs:

```yaml
- !Component
  name: MACC
  class: sc_mac
  attributes:
    mag_bits: 7          # T = 2**mag_bits
    datawidth: 8         # mag_bits + sign
    n_sng_per_mac: 1.0   # dataflow-dependent SNG reuse (see table above)
    # technology + global_cycle_seconds are inherited from the System container
```

`technology` (→ tech node) is inherited from the `System` container's
attributes; `global_cycle_seconds` is passed to `leak()` automatically by
Accelergy. To have `find_mapping.py` emit this automatically, update
`build_arch_spec()` to set `class: sc_mac` and the attributes above — ask and I
can wire that in.
