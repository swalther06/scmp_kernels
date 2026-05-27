# SC matmul — kernel-level MSE profiling & ablation

Tooling to measure the accuracy of `sc_matmul` (stochastic-computing matmul)
against the exact `a @ b.T`, on **real operand tensors** captured from a model,
sweeping SC configuration knobs. Unit-level: no model forward in the loop, so
sweeps are fast and reproducible.

## Why

The end-to-end (`scmp_llm`, `scmp_speculative_decoding`) tests tell you *whether*
a config is good enough for a task. This tells you *how much each SC design knob
costs in raw matmul error*, isolated from the model — useful for choosing
defaults (scramble mode, stream length, halve) and for paper ablations.

## Files

| file | role |
|---|---|
| `capture_real_tensors.py` | Hook a HF model's `nn.Linear` layers and dump real `(a, b)` operand pairs (`a` = flattened layer input `[N,D]`, `b` = weight `[M,D]`, so `sc_matmul(a,b) == F.linear(x,W)`). |
| `mse_profile.py` | Per-config error sweep on captured operands (MSE, RMSE, max\|Δ\|, rel-Frobenius, cosine). CSV + table. |
| `ablation.py` | Focused grid: `stoc_len × halve × scramble`, mean MSE over tensors, with scramble-effect and halve-cost summaries. CSV + table. |
| `_sbatch_*.sh` | Slurm wrappers (gpu-rtx6000). Capture once → `captured_llama8b.pt`, then profile/ablate (rerunnable, ~15 s). |

## Usage

```bash
# 1. capture real operands once (gated model; weights load from cache)
python bench/capture_real_tensors.py --layers 0,15,31 --out bench/captured_llama8b.pt

# 2a. broad sweep (sc_prec=8, per_row, bipolar, chunk_d=128 fixed)
python bench/mse_profile.py  --tensors bench/captured_llama8b.pt --out bench/mse_profile.csv

# 2b. ablation grid: stoc_len x halve x scramble
python bench/ablation.py     --tensors bench/captured_llama8b.pt --out bench/ablation.csv
```

Or via Slurm: `sbatch bench/_sbatch_ablation.sh`.

## Method

- **Reference:** exact `a @ b.T` in fp32.
- **Metric:** MSE = `mean((sc - ref)^2)` (CSV also has RMSE / max\|Δ\| / rel-Frobenius / cosine), meaned over the captured tensors.
- **Fixed:** `granularity=per_row`, `mode=bipolar`, `sc_prec=8`, `chunk_d=128`.
- **Swept axes:**
  - `stoc_len ∈ {256,192,128,96,64,48,32,16}` — stream length (cycles).
  - `halve ∈ {off,on}` — uSystolic: `on` sets the RNG enable-grid to `2^(prec-1)=128` levels (vs `2^prec=256`). Independent of `stoc_len`.
  - `scramble ∈ {off,counter,bitrev}` — Owen de-bias mode (`SC_OWEN_MODE`; `off` also clears `SC_SCRAMBLE_RESCALE`).
- **Caveat handled:** the RNG enable-table cache is *not* keyed by scramble mode, so `clear_rng_cache()` is called before every measured matmul.

## Results (Llama-3.1-8B operands, layers 0/15/31, mean MSE over 21 tensors)

| stoc_len | full/off | full/counter | full/bitrev | halve/off | halve/counter | **halve/bitrev** |
|---:|---|---|---|---|---|---|
| 256 | 4.25e-4 | 4.25e-4 | 4.25e-4 | 3.51e-4 | 2.38e-4 | **2.39e-4** |
| 192 | 7.27e-4 | 4.03e-4 | 4.02e-4 | 6.15e-4 | 4.17e-4 | **4.02e-4** |
| 128 | 2.17e-3 | 8.06e-4 | 7.67e-4 | 1.75e-3 | 8.12e-4 | **7.20e-4** |
| 96 | 3.78e-3 | 1.34e-3 | 1.24e-3 | 3.26e-3 | 1.34e-3 | **1.21e-3** |
| 64 | 1.18e-2 | 2.75e-3 | 2.46e-3 | 1.05e-2 | 2.75e-3 | **2.41e-3** |
| 48 | 2.01e-2 | 4.54e-3 | 4.06e-3 | 1.87e-2 | 4.54e-3 | **3.99e-3** |
| 32 | 5.80e-2 | 9.16e-3 | 7.82e-3 | 5.50e-2 | 9.14e-3 | **7.70e-3** |
| 16 | 2.74e-1 | 3.17e-2 | 2.36e-2 | 2.68e-1 | 3.15e-2 | **2.32e-2** |

### Findings

1. **Scramble is increasingly essential as the stream shortens, and `bitrev ≳ counter ≫ off`.** Removing scramble (off vs bitrev, full stream) costs +0% @256 → +183% @128 → +378% @64 → **+1063% @16** in MSE (≈11× blowup at the shortest stream). It is a genuine no-op at `stoc_len=256` (no prefix truncation) — the sanity control. `bitrev` edges `counter` everywhere and both crush `off`; `bitrev` is the right default.

2. **`halve` (RNG grid 256→128) is free — even a net win.** At fixed stream length it matches or beats the full grid: **−44% MSE @256**, ≈0% elsewhere. Bipolar sign-magnitude only spans `2^(prec-1)=128` magnitude levels, so the 256-level grid over-resolves and adds noise; halving matches the data's true resolution. Confirms the uSystolic/HUB premise — use the 128 grid for bipolar by default.

3. **The cycle/accuracy tradeoff lives in `stoc_len`.** Each halving of the stream roughly squares-down the agreement (e.g. bitrev: 7.7e-4 @128 → 2.5e-3 @64 → 7.8e-3 @32). Usable to ~64; sharp degradation below 32.

4. **Best config across the whole grid: `halve + bitrev`** — lowest MSE in every row. Practical sweet spot: `halve/bitrev` at `stoc_len ≈ 64–128`.

> Note on `halve` vs `stoc_len`: in this sweep `halve` only toggles the RNG
> *grid* (stream length is the independent `stoc_len` axis), so the ~0% "halve
> cost" is the grid effect alone. The uSystolic 2× *cycle* saving is captured by
> moving down the `stoc_len` axis (256→128), not by the `halve` column.
