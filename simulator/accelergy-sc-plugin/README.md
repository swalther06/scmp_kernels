# accelergy-sc-plugin

Accelergy estimators for the **PaYN stochastic-computing (SC) GEMM array**,
modelled as **three components with three different action drivers** — so the
mapper sees each energy term where it physically belongs:

| estimator (file) | class / subclass | Timeloop level | action driver | maps to RTL |
|---|---|---|---|---|
| `SCMacInner` (`sc_mac.py`) | `sc_mac_inner` (compute) | arithmetic | per **MAC** (`compute`) | `InnerPE` / `InnerTile` |
| `Peripheral` (`peripheral.py`) | `storage`/`peripheral` | operand path | per **operand read** | `sc_pe_peripheral` |
| `SobolBank` (`sobol_bank.py`) | `storage`/`sobol_bank` | shared, `instances:1` | per **cycle** (`leak`) | `sobol_bank` ×2 |

> **Status: skeleton.** `sc_mac_inner.compute` carries the **measured** inner
> energy (3.543 pJ/MAC). `peripheral.read` and `sobol_bank.leak` are
> **placeholders** (`E_READ_J` / `E_LEAK_PER_CYCLE_J = None` → per-bit guesses)
> pending the `power_payn_array` PrimeTime run. Good for *relative* mapping
> comparison until then.

## Why three drivers (not one lumped "outer")

The banks and peripheral scale with fundamentally different things — lumping
them mis-models the mapping dependence:

- **Inner** (`sc_mac_inner`) → **per MAC**. Length-T stream scaling lives inside
  the per-`compute` energy (Timeloop counts logical MACs, not stream bits), the
  analogue of `apply_hardware_timing()` for cycles.
- **Peripheral** → **per operand read**. Operands are *reused* (`a[h,k]` feeds a
  whole output row), so conversions ≪ MACs, and reuse changes with the mapping.
  This is the term that actually *moves the energy-optimal mapping*. It's a real
  level in the operand path (keeps Inputs/Weights, bypasses Outputs).
- **Sobol banks** → **per cycle, shared**. One RNG advance broadcasts to the
  whole array, so bank energy is a fixed per-cycle cost independent of operand
  count / MACs / array size. That doesn't fit a data level, so the banks bypass
  all dataspaces and their per-cycle dynamic energy rides in `leak`
  (`instances:1`). Verified: at 256³ this amortizes to ~0.024 pJ/MAC — which is
  exactly *why sharing the banks is the architectural win*.

## Characterizing the placeholders (next step)

Run PrimeTime on PaYN's `power_payn_array` bench and bucket by instance:

```
u_pe                  -> E_inner  ÷ (N_H·N_W·K) MACs      -> sc_mac.E_COMPUTE_J
u_peripheral          -> E_periph ÷ (N_H+N_W)·K operands  -> peripheral.E_READ_J
u_a_rng + u_w_rng     -> E_bank   ÷ window cycles          -> sobol_bank.E_LEAK_PER_CYCLE_J
```
(cross-check `u_pe` pJ/MAC vs the standalone `power_inner_pe` run).

## Install

```bash
pip3 install .                 # -> share/accelergy/estimation_plug_ins/
```

`find_mapping.build_arch_spec()` already emits all three components (`MACC`,
`Peripheral`, `SobolBank`); `make mapping` runs the whole flow in the container.
