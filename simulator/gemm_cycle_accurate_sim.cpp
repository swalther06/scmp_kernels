// Sign-magnitude bipolar stochastic-computing systolic-array simulator.
// Computes O = A*B in one window (matches scmp_kernels sc_matmul: A @ B^T,
// no bias).
//
// Number representation (per scalar) -- matches scmp_kernels bipolar quant
//   Each signed operand is split into:
//     * magnitude : quantized to q_max = MAG_MAX = 2^MAG_BITS - 1, then
//                   REMAPPED onto the RNG grid: boundary = round(mag * RNG_LEVELS / q_max)
//                   where RNG_LEVELS = 2^SC_PREC, SC_PREC = MAG_BITS + 1.
//     * sign      : ONE bit (bool, true = negative)
//   The boundary becomes a stochastic bitstream of length T: each cycle a
//   comparator emits (boundary > rng_threshold), rng in [0, RNG_LEVELS), so
//   the fraction of 1s ~= boundary / RNG_LEVELS ~= mag / q_max -- exactly
//   scmp's convention.  Only magnitudes are streamed; signs never become
//   bitstreams.  Product sign = sign_a XNOR sign_b.
//
// Tile inner structure (lower tier of the hierarchy)
//   * K diagonal AND gates form the magnitude dot product on stream bits.
//   * K XNOR gates over the per-tile stationary sign registers give each
//     lane's product sign.  These sign registers change at most ONCE per
//     window (a whole bitstream shares one sign), so the XNOR output is
//     constant for all T cycles -- no per-cycle sign switching.
//   * Each tick processes M threshold samples per lane, so the popcount spans
//     K*M bits.  Adding K*M + popcnt_pos - popcnt_neg (always in [0, 2K*M])
//     into a SINGLE accumulator oreg keeps the value non-negative (the +K*M
//     bias) while using one register and one adder per PE.
//   * Window length is T/M ticks (M samples consumed per tick).
//   * drain_reg = oreg - K*T : the bias K*M accumulated over T/M ticks equals
//     K*M*(T/M) = K*T, removed ONCE as a compile-time CONSTANT at drain.
//
// Systolic geometry (P_ROWS x P_COLS tiles, may be non-square)
//   inputs  flow west  -> east   (west column is the input  edge)
//   weights flow south -> north  (south row  is the weight  edge)
//   TRUE systolic propagation: a stream bit generated at an edge tile is
//   latched into the inter-tile flop and read by the neighbor ONE cycle
//   later, marching one tile per cycle (link_we_mag / link_ns_mag toggle
//   every cycle).  Signs ride the same path but, being constant for a whole
//   bitstream, their flops (link_we_sign / link_ns_sign) update only once
//   per window -- counted by sign_link_toggle, not link_toggle.
//
//   Warm-up: bits march one tile per cycle, so interior tile (r,c) sees
//   nothing on its input port for the first c cycles nor on its weight port
//   for the first (P_ROWS-1-r) cycles (the links start at 0).  This is real
//   hardware latency; we keep it.  It costs <= (array side - 1) of T cycles,
//   so it matters more at short T -- run extra pipeline-fill cycles if needed.
//
// Decode / dequant (matches scmp's net per-term scale)
//   A lane's expected AND count over T cycles is
//     T * (boundary_a / RNG_LEVELS) * (boundary_b / RNG_LEVELS)
//       ~= T * (mag_a / q_max) * (mag_w / q_max),
//   so  drain = oreg - K*T ~= (sum mag_a*mag_w) * T / q_max^2.
//   With per-tensor scale S = abs_max / q_max, the real dot product is
//     real = drain * S_a * S_b / T
//   which is exactly scmp's decode count*(q_max^2/stoc_len)*(S_a)*(S_b)
//   after substituting S = abs_max/q_max.  No SC_DENOM, no bias.
//
// Lane decorrelation (CFG_OWEN, default off)
//   Only M physical Sobol generators exist per side (array-wide), so all K
//   dot-product lanes -- and, within a tile, all N_H/N_W rows -- share the
//   same M rng thresholds per cycle (the cheap "amortized RNG" choice). Left
//   unscrambled, the K lanes summed into one accumulator are correlated and
//   the dot product forfeits the 1/sqrt(K) error averaging it would get from
//   independent per-lane randomness.  CFG_OWEN=1 XORs each (K,M) lane's copy
//   of the shared threshold with a distinct stationary mask built from two
//   Weyl/golden-ratio strides (one per axis) -- a per-lane wire tap, no extra
//   RNG -- which partially restores that averaging. Bit-exact port of
//   designs/sc_gemm_3d/cmp_lane.sv in the sibling SCArch repo. See make_owen()
//   and OWEN_IN / OWEN_W.
//
// Energy accounting -- what is and isn't counted (see struct Stats):
//   counted    : AND-1 events, accumulator updates, mag wire toggles, the
//                once-per-window sign wire toggles, drain shifts, edge loads,
//                RNG steps, comparator evals.  The single-accumulator +K
//                offset (see "Tile inner structure" above) means oreg updates
//                almost every cycle, even when all K lanes AND to 0 -- an
//                inherent cost of folding the old +/- split into one register.
//   NOT counted: the stationary sign XNOR (no per-cycle switching),
//                popcount-tree internal adds (only the net accumulator
//                increment is priced).
//
// Build: `make` (see Makefile), or directly
//   g++ -std=c++17 -O2 -Wall -Wextra tile.cpp -o tile_sim
// Override knobs at compile time, e.g.:  make CFG="-DCFG_K=8 -DCFG_OWEN=1"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <iostream>
#include <string>
#include <vector>

// ---- Tunable parameters --------------------------------------------------
// Every knob has a default below and can be overridden at COMPILE time, e.g.:
//   g++ -std=c++17 -O2 -DCFG_MAG_BITS=8 -DCFG_K=8 -DCFG_P_ROWS=2 -DCFG_P_COLS=6 tile.cpp
// The CFG_* macros are just the override hooks; the typed constexpr constants
// below are what the rest of the code uses.
#ifndef CFG_MAG_BITS
#define CFG_MAG_BITS 7                  // magnitude bit width (precision)
#endif
#ifndef CFG_K
#define CFG_K 4                         // diagonal length per tile (dot-product depth)
#endif
#ifndef CFG_P_ROWS
#define CFG_P_ROWS 4                    // systolic tile rows (max output rows / window)
#endif
#ifndef CFG_P_COLS
#define CFG_P_COLS 4                    // systolic tile cols (max output cols / window)
#endif
#ifndef CFG_STOC_LEN
#define CFG_STOC_LEN (1 << CFG_MAG_BITS) // bitstream length T; default 2^MAG_BITS = 2^(sc_prec-1)
#endif
#ifndef CFG_OWEN
#define CFG_OWEN 0                       // 1 = per-lane Owen XOR scramble (decorrelate K lanes)
#endif
#ifndef CFG_N_H
#define CFG_N_H 1                     // PEs per tile, row direction (M tier height); >1 not yet implemented
#endif
#ifndef CFG_N_W
#define CFG_N_W 1                     // PEs per tile, column direction (M tier width); >1 not yet implemented
#endif
#ifndef CFG_M
#define CFG_M 1                          // stream bits processed per cycle (L); >1 not yet implemented
#endif

// Precision model matches scmp_kernels bipolar quantization:
//   sc_prec    = MAG_BITS + 1            (e.g. 8-bit signed = 7 mag bits + 1 sign)
//   q_max      = 2^(sc_prec-1) - 1 = MAG_MAX   (largest magnitude)
//   RNG grid   = 2^sc_prec = RNG_LEVELS        (comparator/RNG are sc_prec bits wide)
// A magnitude is REMAPPED onto the RNG grid before comparison (see quantize):
//   boundary = round(mag * RNG_LEVELS / q_max),  compared as boundary > rng.
// This gives P(bit) = boundary/RNG_LEVELS ~= mag/q_max, exactly like scmp.
constexpr int   MAG_BITS   = CFG_MAG_BITS;                  // magnitude bit width
using           mag_t      = uint32_t;                      // holds boundary/threshold (<=31 bits)
constexpr mag_t MAG_MAX    = (mag_t(1) << MAG_BITS) - 1;    // q_max = largest magnitude (e.g. 127)
constexpr int   SC_PREC    = MAG_BITS + 1;                  // full SC precision (e.g. 8)
constexpr mag_t RNG_LEVELS = mag_t(1) << SC_PREC;           // RNG/comparator grid (e.g. 256)
constexpr mag_t RNG_MASK   = RNG_LEVELS - 1;                // rng threshold mask (0..RNG_LEVELS-1)
constexpr int   T          = CFG_STOC_LEN;                  // bitstream length (cycles per window)
constexpr int   K          = CFG_K;                         // diagonal length per tile
constexpr int   P_ROWS     = CFG_P_ROWS;                    // systolic tile rows
constexpr int   P_COLS     = CFG_P_COLS;                    // systolic tile cols
constexpr int   N_H     = CFG_N_H;                    // PEs per tile, rows (combinational tier height)
constexpr int   N_W     = CFG_N_W;                    // PEs per tile, cols (combinational tier width)
constexpr int   M          = CFG_M;                         // stream bits per cycle (bitstream parallelism)

// All four DSE axes are implemented:
//   P (P_ROWS x P_COLS): systolic tile grid; N dimension in RTL (pe_outer grid).
//   N (N_H x N_W):       combinational PE block per tile; M dimension in RTL (inner_tile grid).
//   K:                   dot-product lane width.
//   M:                   bitstream parallelism -- M threshold samples per lane per tick,
//                        so the window finishes in T/M ticks.  Each tick's popcount spans
//                        K*M bits (all K lanes x M samples); a single accumulator per PE
//                        collects K*M + popcnt_pos - popcnt_neg each tick.
static_assert(T % M == 0, "CFG_M must divide T (bitstream length)");

// Activity counters used as proxies for dynamic energy.
struct Stats {
    uint64_t and_ones         = 0;  // AND outputs that were 1 (AND-array switching)
    uint64_t and_pos          = 0;  // AND-1 events routed to popcnt_pos
    uint64_t and_neg          = 0;  // AND-1 events routed to popcnt_neg
    uint64_t oreg_inc         = 0;  // non-zero accumulator updates (either polarity)
    uint64_t add_op           = 0;  // adds  (pos accum, neg accum)
    uint64_t sub_op           = 0;  // subs  (oreg - K*T at drain)
    uint64_t link_toggle      = 0;  // inter-tile MAG flop bit transitions (per cycle)
    uint64_t sign_link_toggle = 0;  // inter-tile SIGN flop transitions (once per window per hop)
    uint64_t drain_shift      = 0;  // drain register transitions during shift
    uint64_t bin_load         = 0;  // edge binary-register loads (mag+sign bits at the edges)
    uint64_t rng_advance      = 0;  // Sobol steps
    uint64_t cmp_eval         = 0;  // edge comparator evaluations
    void add(const Stats& s) {
        and_ones += s.and_ones; and_pos += s.and_pos; and_neg += s.and_neg;
        oreg_inc += s.oreg_inc; add_op += s.add_op;   sub_op += s.sub_op;
        link_toggle += s.link_toggle; sign_link_toggle += s.sign_link_toggle;
        drain_shift += s.drain_shift; bin_load += s.bin_load;
        rng_advance += s.rng_advance; cmp_eval += s.cmp_eval;
    }
};

// scmp_kernels-compatible Sobol direction vectors (sc/rng.py).
//   SCGen formula:  V[i] = floor(seed[i] / 2^(i+1) * 2^SC_PREC)
//                        = (seed[i] << SC_PREC) >> (i+1)   (exact integer)
//   Q (inputs)  uses seed [1,1,...,1]               -> V = [128,64,32,...] (sc_prec=8)
//   K (weights) uses the low-SCC seed from _default_seed("k"); for sc_prec=8 that
//   is [1,1,1,1,9,1,41,255] (SCC~=0 vs Q), else the fallback taps seed[4]=9, seed[6]=41.
constexpr std::array<int, SC_PREC> sobol_seed_q() {
    std::array<int, SC_PREC> s{};
    for (int i = 0; i < SC_PREC; ++i) s[i] = 1;
    return s;
}
constexpr std::array<int, SC_PREC> sobol_seed_k() {
    std::array<int, SC_PREC> s{};
    for (int i = 0; i < SC_PREC; ++i) s[i] = 1;
    if constexpr (SC_PREC == 8) {
        const int kv[8] = {1, 1, 1, 1, 9, 1, 41, 255};
        for (int i = 0; i < 8; ++i) s[i] = kv[i];
    } else {
        if (SC_PREC >= 5) s[4] = 9;
        if (SC_PREC >= 7) s[6] = 41;
    }
    return s;
}
constexpr std::array<mag_t, SC_PREC> sobol_dirvec(const std::array<int, SC_PREC>& seed) {
    std::array<mag_t, SC_PREC> v{};
    for (int i = 0; i < SC_PREC; ++i)
        v[i] = mag_t((uint64_t(seed[i]) << SC_PREC) >> (i + 1));
    return v;
}
constexpr std::array<mag_t, SC_PREC> SOBOL_V1 = sobol_dirvec(sobol_seed_q());
constexpr std::array<mag_t, SC_PREC> SOBOL_V2 = sobol_dirvec(sobol_seed_k());

// Per-(K,M)-lane Owen XOR masks (only active when CFG_OWEN=1).  Bit-exact port
// of designs/sc_gemm_3d/cmp_lane.sv: a single M-wide threshold bus is shared
// across ALL K dot-product lanes (only M physical Sobol generators exist per
// side, broadcast array-wide -- see rng_bank.sv / sc_array.sv), so without a
// per-(d,m) mask every K lane would compare against the identical threshold
// every cycle. XOR-ing each (d,m) lane's threshold with a distinct stationary
// mask decorrelates them (a bijection on [0,RNG_LEVELS), so each lane's
// marginal -- and thus its encoded magnitude -- is unchanged). Two Weyl/
// golden-ratio strides (exact integer form, matching the RTL's synthesizable
// (LEVELS*79/128) / (LEVELS*49/128), NOT a floating-point golden ratio) spread
// the K x M lane grid evenly across the threshold domain; weights use a
// half-step salt offset so input and weight lane-masks differ too. This is a
// wire tap, no extra RNG -- the same trick as scmp_kernels' _owen_scramble.
constexpr mag_t GOLDEN_K = mag_t(((uint64_t(RNG_LEVELS) * 79) / 128) | 1);
constexpr mag_t GOLDEN_M = mag_t(((uint64_t(RNG_LEVELS) * 49) / 128) | 1);
constexpr std::array<std::array<mag_t, M>, K> make_owen(mag_t salt) {
    std::array<std::array<mag_t, M>, K> mask{};
    for (int d = 0; d < K; ++d)
        for (int m = 0; m < M; ++m)
            mask[d][m] = CFG_OWEN
                ? mag_t((mag_t(d) * GOLDEN_K + mag_t(m) * GOLDEN_M + salt) & RNG_MASK)
                : mag_t(0);
    return mask;
}
constexpr std::array<std::array<mag_t, M>, K> OWEN_IN = make_owen(0);
constexpr std::array<std::array<mag_t, M>, K> OWEN_W  = make_owen(RNG_LEVELS >> 1);

// scmp_kernels-compatible Sobol stepping (sc/rng.py _step): Gray-code XOR that
// returns the value BEFORE applying the XOR, so the emitted sequence starts at
// 0 and, over a full 2^SC_PREC period, is a permutation of [0, 2^SC_PREC).
// scmp uses lsz(index) (least-significant ZERO of the pre-increment index),
// which equals ctz(index+1).  When the tap runs off the end (k >= SC_PREC) the
// value is left unchanged -- exactly as rng.py (no forced wrap). Verified
// bit-exact vs scmp over the full 256-value Q and K periods.
struct Sobol {
    mag_t    value = 0;
    uint64_t index = 0;
    const mag_t* V = SOBOL_V1.data();

    explicit Sobol(const mag_t* dirvec = SOBOL_V1.data()) : V(dirvec) {}

    mag_t step() {
        mag_t out = value;                    // return-before-xor (matches rng.py)
        int k = __builtin_ctzll(index + 1);   // lsz(index) == ctz(index+1)
        ++index;
        if (k < SC_PREC) value ^= V[k];
        return out & RNG_MASK;
    }
};

// M-wide Sobol bank -- bit-exact port of designs/sc_gemm_3d/rng_bank.sv /
// sc_kernel.py's RNGBank in the sibling SCArch repo. rng_bank.sv does NOT
// share one Sobol generator M ways; it instantiates M independent generators,
// each seeded SEED_BASE ^ (SEED_STRIDE * m), all driven by the same DV table
// in lockstep -- so lane m's threshold at cycle t is seed_m XOR V(t), not the
// (t + m)'th term of one running sequence. These are the literal seed/stride
// pairs from sc_array.sv's u_rng_in / u_rng_w instances (WIDTH=8 8-bit
// literals -- only exactly matches the RTL when MAG_BITS=7/SC_PREC=8, same
// restriction sc_array.sv itself has).
constexpr mag_t SEED_BASE_IN   = 0x17;
constexpr mag_t SEED_STRIDE_IN = 0x53;
constexpr mag_t SEED_BASE_W    = 0x9D;
constexpr mag_t SEED_STRIDE_W  = 0x2B;

// sobol.sv's sobolSeq is a REGISTERED output: it reads SEED at cycle 0 and
// becomes sobolSeq ^ selected_dv(cnt) *before* being read the next cycle
// ("return-after"). struct Sobol above instead returns the value *before*
// applying that cycle's XOR ("return-before", matching scmp_kernels' rng.py,
// which has no seed-as-initial-value concept at all). Algebraically these are
// the same sequence shifted by exactly one step, so seeding a lane with
// `value = seed` and then discarding one throwaway step() call reproduces the
// RTL's cycle-1 value on the next real call.
std::array<Sobol, M> make_sobol_bank(mag_t seed_base, mag_t seed_stride, const mag_t* dirvec) {
    std::array<Sobol, M> bank;
    for (int m = 0; m < M; ++m) {
        bank[m].V     = dirvec;
        bank[m].value = mag_t((seed_base ^ mag_t((uint64_t(seed_stride) * m) & RNG_MASK)) & RNG_MASK);
        bank[m].index = 0;
        bank[m].step();   // prime: align phase with the RTL's registered SEED
    }
    return bank;
}

// Tile: an N_H x N_W combinational block of PEs (the N tier).  Each tick,
// PE(i,j) evaluates K*M AND gates (K lanes x M parallel threshold samples)
// then adds K*M + popcnt_pos - popcnt_neg (in [0, 2K*M]) into oreg[i][j].
// Row i's input K-vector broadcasts across all N_W PEs in that row; column
// j's weight K-vector broadcasts across all N_H PEs in that column.  No
// registers between PEs (combinational within a tile; systolic flops live
// between tiles at the Systolic level).
class Tile {
public:
    bool is_input_edge  = false;   // west column: comparator drives in_mag_bits from in_mag
    bool is_weight_edge = false;   // south row : comparator drives w_mag_bits  from w_mag

    // Stationary operand registers (held for a whole window): N_H distinct
    // input K-vectors, N_W distinct weight K-vectors.
    // in_mag/w_mag hold the remapped boundary (0..RNG_LEVELS), not raw magnitude.
    std::array<std::array<mag_t, K>, N_H> in_mag{};
    std::array<std::array<mag_t, K>, N_W> w_mag{};
    std::array<std::array<bool,  K>, N_H> in_sign{};  // 1-bit signs (true = negative)
    std::array<std::array<bool,  K>, N_W> w_sign{};

    // Per-tick stream bits -- MAGNITUDE only (signs are stationary, above).
    // Third dimension is M: M parallel threshold samples per (lane, K-index).
    std::array<std::array<std::array<bool, M>, K>, N_H> in_mag_bits{};
    std::array<std::array<std::array<bool, M>, K>, N_W> w_mag_bits{};
    // Pass-through bits latched into downstream inter-tile flops.
    std::array<std::array<std::array<bool, M>, K>, N_H> in_mag_out{};
    std::array<std::array<std::array<bool, M>, K>, N_W> w_mag_out{};

    // Single biased accumulator per PE (replaces oreg_pos/oreg_neg): holds
    // sum over the window of (K + popcnt_pos - popcnt_neg), i.e. the true
    // dot product plus a K*T offset removed once at drain (see load_drain).
    std::array<std::array<int, N_W>, N_H> oreg{};
    std::array<std::array<int, N_W>, N_H> drain_reg{};
    Stats stats;

    // One tick.  Edge faces generate M comparisons per lane from M thresholds;
    // non-edge faces were driven by Systolic from the link flop latched last tick.
    void compute(const std::array<mag_t, M>& rng_i, const std::array<mag_t, M>& rng_w) {
        if (is_input_edge) {
            for (int i = 0; i < N_H; ++i)
                for (int d = 0; d < K; ++d)
                    for (int m = 0; m < M; ++m) {
                        mag_t thr_i = (rng_i[m] ^ OWEN_IN[d][m]) & RNG_MASK;
                        in_mag_bits[i][d][m] = in_mag[i][d] > thr_i;
                        ++stats.cmp_eval;
                    }
        }
        if (is_weight_edge) {
            for (int j = 0; j < N_W; ++j)
                for (int d = 0; d < K; ++d)
                    for (int m = 0; m < M; ++m) {
                        mag_t thr_w = (rng_w[m] ^ OWEN_W[d][m]) & RNG_MASK;
                        w_mag_bits[j][d][m] = w_mag[j][d] > thr_w;
                        ++stats.cmp_eval;
                    }
        }
        // N_H x N_W PEs: row i's input K*M bits feed every column, column
        // j's weight K*M bits feed every row; sign is per-lane (per d), constant
        // for the whole window, so the XNOR is stationary (no per-tick switching).
        for (int i = 0; i < N_H; ++i) {
            for (int j = 0; j < N_W; ++j) {
                int popcnt_pos = 0, popcnt_neg = 0;
                for (int d = 0; d < K; ++d) {
                    bool neg = (in_sign[i][d] != w_sign[j][d]);
                    for (int m = 0; m < M; ++m) {
                        bool a = in_mag_bits[i][d][m] && w_mag_bits[j][d][m];
                        if (a) {
                            if (neg) { ++popcnt_neg; ++stats.and_neg; }
                            else     { ++popcnt_pos; ++stats.and_pos; }
                            ++stats.and_ones;
                        }
                    }
                }
                // K*M bias keeps oreg non-negative (range [0, 2K*M] per tick).
                int delta = K*M + popcnt_pos - popcnt_neg;
                if (delta) { oreg[i][j] += delta; ++stats.oreg_inc; ++stats.add_op; }
            }
        }
        in_mag_out = in_mag_bits;
        w_mag_out  = w_mag_bits;
    }

    // Drain combine, per PE: remove the K*T offset built up over the window
    // by subtracting a compile-time CONSTANT (no bias term in the GEMM
    // itself; sc_matmul computes A*B only). `windows` is how many T/M-tick
    // windows accumulated into oreg since the last drain -- >1 when multiple
    // K-blocks were chained into one accumulate window without an
    // intervening drain (see Systolic::end_window / run_workload_binary's
    // multi-K-block loop): each window adds another K*M*(T/M) = K*T of bias,
    // so removing it in one shot at the end needs K*T*windows, not K*T. This
    // is equivalent to sc_kernel.py's per-cycle bias subtract (K*M every
    // cycle, summed continuously across K-blocks into one accumulator)
    // because the bias is linear -- subtracting the same total amount once
    // at the end gives an identical final result.
    void load_drain(int windows = 1) {
        for (int i = 0; i < N_H; ++i)
            for (int j = 0; j < N_W; ++j) {
                drain_reg[i][j] = oreg[i][j] - K * T * windows;
                ++stats.sub_op;   // oreg - K*T*windows
                oreg[i][j] = 0;
            }
    }

    // Drain shift, one column-position per cycle: lane i is an independent
    // east-going port (N_H PE-rows never share hardware, same as on the
    // compute side), but the N_W PE-columns within that lane DO share one
    // port and so must serialize through it.  Lane i's front value (j=0)
    // exits east -- returned in 'sent', picked up by the east neighbor's
    // shift_drain call, or by Systolic's east_out at the east edge; positions
    // 1..N_W-1 shift toward the front; the back (j=N_W-1) is refilled
    // by from_west[i], the value the WEST neighbor is sending this very
    // cycle (0 once that tile's own data is exhausted, or for the west-most
    // tile, which has nothing further west to draw from).  Draining one
    // tile's N_W values takes N_W cycles; P_COLS tiles in series take
    // P_COLS*N_W, matching the file's one-hop-per-cycle systolic style.
    std::array<int, N_H> shift_drain(const std::array<int, N_H>& from_west) {
        std::array<int, N_H> sent{};
        for (int i = 0; i < N_H; ++i) {
            sent[i] = drain_reg[i][0];
            for (int j = 0; j + 1 < N_W; ++j) {
                if (drain_reg[i][j] != drain_reg[i][j + 1]) ++stats.drain_shift;
                drain_reg[i][j] = drain_reg[i][j + 1];
            }
            if (drain_reg[i][N_W - 1] != from_west[i]) ++stats.drain_shift;
            drain_reg[i][N_W - 1] = from_west[i];
        }
        return sent;
    }

    // Clear transient per-window state. Stationary operands and Stats kept.
    void reset_window() {
        for (auto& row : oreg)        row.fill(0);
        for (auto& row : drain_reg)   row.fill(0);
        for (auto& row : in_mag_bits) for (auto& bits : row) bits.fill(false);
        for (auto& row : w_mag_bits)  for (auto& bits : row) bits.fill(false);
        for (auto& row : in_mag_out)  for (auto& bits : row) bits.fill(false);
        for (auto& row : w_mag_out)   for (auto& bits : row) bits.fill(false);
    }
};

// Systolic: owns the two array-wide Sobols, the inter-tile flops (mag + sign
// halves, N_H or N_W vectors of 1 bit per lane), and the tile grid.
//   tiles[0][.]          top row
//   tiles[P_ROWS-1][.]   south (weight) edge
//   tiles[.][0]          west (input) edge
//   tiles[P_ROWS-1][0]   corner (does both conversions)
class Systolic {
public:
    std::array<Sobol, M> rng_input, rng_weight;
    std::array<std::array<Tile, P_COLS>, P_ROWS> tiles;
    // MAGNITUDE bitstream flops (toggle every cycle).  link_*[r][c] feeds tile
    // (r,c) the same N_H (input) or N_W (weight) lane vectors its Tile
    // holds internally.
    std::array<std::array<std::array<std::array<std::array<bool, M>, K>, N_H>, P_COLS>, P_ROWS> link_we_mag{};
    std::array<std::array<std::array<std::array<std::array<bool, M>, K>, N_W>, P_COLS>, P_ROWS> link_ns_mag{};
    // SIGN flops (1 bit per lane).  Written once per window by load_*; held all T cycles.
    std::array<std::array<std::array<std::array<bool, K>, N_H>, P_COLS>, P_ROWS> link_we_sign{};
    std::array<std::array<std::array<std::array<bool, K>, N_W>, P_COLS>, P_ROWS> link_ns_sign{};
    std::array<int, P_ROWS * N_H> east_out{};   // indexed by global row r*N_H+i
    Stats systolic_stats;
    int  tick_in_phase = 0;
    bool computing     = true;
    uint64_t cycle     = 0;

    Systolic(const mag_t* dirvec_i = SOBOL_V1.data(),
             const mag_t* dirvec_w = SOBOL_V2.data())
        : rng_input(make_sobol_bank(SEED_BASE_IN, SEED_STRIDE_IN, dirvec_i)),
          rng_weight(make_sobol_bank(SEED_BASE_W, SEED_STRIDE_W, dirvec_w)) {
        for (int r = 0; r < P_ROWS; ++r) tiles[r][0].is_input_edge         = true;
        for (int c = 0; c < P_COLS; ++c) tiles[P_ROWS-1][c].is_weight_edge = true;
    }

    // Load A's global row (0..P_ROWS*N_H-1) into internal PE-row i of
    // west-edge tile r, where r = row/N_H, i = row%N_H -- i.e. tile
    // row r's N_H internal PE-rows are global rows [r*N_H, r*N_H+N_H).
    // Magnitude becomes the bitstream via the comparator and marches east one
    // tile per cycle through link_we_mag[.][.][i]; all N_W PEs in PE-row i
    // see the same stream (input reused across columns, per the tile doc).
    // Sign is constant for the whole stream, so it is set into each tile's
    // sign register and the sign flops; those flops change at most once per
    // window (sign_link_toggle counts only real changes).
    void load_inputs(int row, const std::array<mag_t, K>& mag,
                              const std::array<bool,  K>& sign) {
        int r = row / N_H, i = row % N_H;
        tiles[r][0].in_mag[i]  = mag;
        tiles[r][0].in_sign[i] = sign;
        tiles[r][0].stats.bin_load += 2 * K;   // edge mag reg + edge sign reg
        for (int c = 1; c < P_COLS; ++c) {
            for (int d = 0; d < K; ++d)
                if (link_we_sign[r][c][i][d] != sign[d]) ++systolic_stats.sign_link_toggle;
            link_we_sign[r][c][i]  = sign;         // one-shot per-window sign hop
            tiles[r][c].in_sign[i] = sign;         // tile's stationary sign register
        }
    }
    // Load B's global col (0..P_COLS*N_W-1) into internal PE-col j of
    // south-edge tile c, where c = col/N_W, j = col%N_W (mirror of the
    // above, weights flow south->north; weight reused across rows).
    void load_weights(int col, const std::array<mag_t, K>& mag,
                               const std::array<bool,  K>& sign) {
        int c = col / N_W, j = col % N_W;
        tiles[P_ROWS-1][c].w_mag[j]  = mag;
        tiles[P_ROWS-1][c].w_sign[j] = sign;
        tiles[P_ROWS-1][c].stats.bin_load += 2 * K;
        for (int r = 0; r < P_ROWS - 1; ++r) {
            for (int d = 0; d < K; ++d)
                if (link_ns_sign[r][c][j][d] != sign[d]) ++systolic_stats.sign_link_toggle;
            link_ns_sign[r][c][j]  = sign;
            tiles[r][c].w_sign[j]  = sign;
        }
    }

    // Clear inter-tile MAG flops and per-tile transient bits (NOT sign flops,
    // NOT oreg/drain -- signs are reloaded each window, oreg handled by drain).
    void clear_window_transient() {
        for (auto& row : link_we_mag) for (auto& cell : row) for (auto& lane : cell) for (auto& bits : lane) bits.fill(false);
        for (auto& row : link_ns_mag) for (auto& cell : row) for (auto& lane : cell) for (auto& bits : lane) bits.fill(false);
        for (auto& row : tiles)
            for (auto& t : row) {
                for (auto& lane : t.in_mag_bits) for (auto& bits : lane) bits.fill(false);
                for (auto& lane : t.w_mag_bits)  for (auto& bits : lane) bits.fill(false);
                for (auto& lane : t.in_mag_out)  for (auto& bits : lane) bits.fill(false);
                for (auto& lane : t.w_mag_out)   for (auto& bits : lane) bits.fill(false);
            }
    }

    void reset_window_state() {
        clear_window_transient();
        for (auto& row : tiles)
            for (auto& t : row) {
                for (auto& lane : t.oreg)      lane.fill(0);
                for (auto& lane : t.drain_reg) lane.fill(0);
            }
        east_out.fill(0);
        tick_in_phase = 0;
        computing     = true;
    }

    // One compute step: RNG advance + systolic hop + tile compute + link
    // latch (steps 1-4 below). Does NOT check for window-end -- callers that
    // need multiple independent accumulate windows before draining (e.g.
    // multiple K-blocks accumulated into one spatial-tile output, mirroring
    // designs/sc_gemm_3d/inner_tile_2bit.sv's `acc` register, which only
    // clears on clr_acc/reset -- never just from running another cycle) call
    // this directly and invoke end_window() themselves once, after all
    // K-blocks. tick() below is the original single-window auto-draining
    // path, built on top of the same two primitives.
    void step_compute() {
        ++cycle;
        // (1) step each of the M input-bank and M weight-bank lanes once,
        //     to get M threshold samples per side (one physical Sobol
        //     generator per lane, not one generator stepped M times).
        std::array<mag_t, M> ri, rw;
        for (int m = 0; m < M; ++m) {
            ri[m] = rng_input[m].step();
            rw[m] = rng_weight[m].step();
            systolic_stats.rng_advance += 2;
        }

        // (2) drive non-edge faces from the inter-tile flop latched LAST
        //     cycle -- this IS the systolic hop (one tile per cycle).
        //     Edge faces are generated inside compute() from the comparator.
        for (int r = 0; r < P_ROWS; ++r)
            for (int c = 0; c < P_COLS; ++c) {
                if (!tiles[r][c].is_input_edge)  tiles[r][c].in_mag_bits = link_we_mag[r][c];
                if (!tiles[r][c].is_weight_edge) tiles[r][c].w_mag_bits  = link_ns_mag[r][c];
            }

        // (3) every tile computes combinationally (reads its own bits only)
        for (int r = 0; r < P_ROWS; ++r)
            for (int c = 0; c < P_COLS; ++c)
                tiles[r][c].compute(ri, rw);

        // (4) latch this cycle's pass-through MAG bits into the downstream
        //     flops (read next cycle).  Signs are stationary -> untouched.
        for (int r = 0; r < P_ROWS; ++r)
            for (int c = 0; c < P_COLS; ++c) {
                if (c + 1 < P_COLS) {                 // input bits hop east
                    for (int i = 0; i < N_H; ++i)
                        for (int d = 0; d < K; ++d)
                            for (int m = 0; m < M; ++m)
                                if (link_we_mag[r][c+1][i][d][m] != tiles[r][c].in_mag_out[i][d][m])
                                    ++systolic_stats.link_toggle;
                    link_we_mag[r][c+1] = tiles[r][c].in_mag_out;
                }
                if (r > 0) {                          // weight bits hop north
                    for (int j = 0; j < N_W; ++j)
                        for (int d = 0; d < K; ++d)
                            for (int m = 0; m < M; ++m)
                                if (link_ns_mag[r-1][c][j][d][m] != tiles[r][c].w_mag_out[j][d][m])
                                    ++systolic_stats.link_toggle;
                    link_ns_mag[r-1][c] = tiles[r][c].w_mag_out;
                }
            }
    }

    // End the current accumulate window: drain each tile's oreg into
    // drain_reg (removing the K*T*windows bias, see Tile::load_drain) and
    // clear transient per-window state. Enters the drain-shift phase
    // (computing = false), exactly as the original inline step 5 did.
    void end_window(int windows = 1) {
        for (auto& row : tiles) for (auto& t : row) t.load_drain(windows);
        clear_window_transient();
        computing     = false;
        tick_in_phase = 0;
    }

    void tick() {
        if (computing) {
            step_compute();
            // end of window after T/M ticks (M samples consumed per tick).
            if (++tick_in_phase == T / M) end_window();
        } else {
            ++cycle;
            // Drain: each tile-row's N_H lanes are independent east ports
            // (fed west-to-east so each shift_drain call gets a same-cycle
            // snapshot of what its west neighbor is sending); within a lane,
            // a tile's N_W columns serialize through that one port.  Total
            // P_COLS*N_W cycles per physical row -- see Tile::shift_drain.
            for (int r = 0; r < P_ROWS; ++r) {
                std::array<int, N_H> carry{};   // 0 = nothing arriving from further west
                for (int c = 0; c < P_COLS; ++c)
                    carry = tiles[r][c].shift_drain(carry);
                for (int i = 0; i < N_H; ++i) east_out[r * N_H + i] = carry[i];
            }
            if (++tick_in_phase == P_COLS * N_W) { computing = true; tick_in_phase = 0; }
        }
    }

    bool draining() const { return !computing; }
    int  east_row(int r) const { return east_out[r]; }

    Stats total_stats() const {
        Stats s = systolic_stats;
        for (auto& row : tiles) for (auto& t : row) s.add(t.stats);
        return s;
    }
};

// ---------------------------------------------------------------------------
// End-to-end FP demo: real matrices -> quantize -> SC array -> dequantize.
// Operands are FIXED real values (independent of MAG_BITS) so a precision
// sweep over MAG_BITS is a fair comparison of accuracy vs bit width.
// ---------------------------------------------------------------------------

// Quantize a real value (|v| <= scale) to (boundary, sign), matching scmp:
//   1. normalize to [-1, 1] and round onto the magnitude grid q in [0, q_max]
//   2. remap onto the RNG grid: boundary = round(q * RNG_LEVELS / q_max)
// The comparator then does boundary > rng (rng in [0, RNG_LEVELS)). Both the
// round() and the remap are software/quantizer steps -- the SC array only ever
// sees the integer boundary.  'mag' returns the boundary.
static void quantize(double v, double scale, mag_t& mag, bool& sign) {
    double n = v / scale;                                          // normalize to [-1, 1]
    long q = long(std::nearbyint(n * double(MAG_MAX)));            // half-to-even (scmp nearbyint)
    if (q < 0) { sign = true; q = -q; } else { sign = false; }
    if (q > long(MAG_MAX)) q = long(MAG_MAX);
    long boundary = long(std::nearbyint(double(q) * double(RNG_LEVELS) / double(MAG_MAX)));
    if (boundary > long(RNG_MASK)) boundary = long(RNG_MASK);      // clamp into [0, RNG_LEVELS)
    mag = mag_t(boundary);
}

static void bipolar_matmul_demo() {
    // OUT_ROWS/OUT_COLS are the GEMM's logical output dims -- the FULL physical
    // grid (P_ROWS*N_H rows, P_COLS*N_W cols), not the architecture's
    // N_H/N_W knobs themselves (see the K-vs-M doc: these are sizing
    // knobs, not a GEMM's logical dimensions).
    constexpr int OUT_ROWS = P_ROWS * N_H, K_mat = K, OUT_COLS = P_COLS * N_W;

    // Fixed real operands (do NOT depend on MAG_BITS). No bias: O = A*B.
    double A_real[OUT_ROWS][K_mat], B_real[K_mat][OUT_COLS];
    for (int m = 0; m < OUT_ROWS; ++m)
        for (int k = 0; k < K_mat; ++k)
            A_real[m][k] = 1.3 * std::sin(0.9 * (m * K_mat + k) + 0.3);
    for (int k = 0; k < K_mat; ++k)
        for (int n = 0; n < OUT_COLS; ++n)
            B_real[k][n] = 0.8 * std::cos(0.7 * (k * OUT_COLS + n) + 0.4);

    // Per-tensor abs-max scales so operands normalize into [-1, 1].
    double S_a = 1e-9, S_b = 1e-9;
    for (int m = 0; m < OUT_ROWS; ++m) for (int k = 0; k < K_mat; ++k) S_a = std::max(S_a, std::fabs(A_real[m][k]));
    for (int k = 0; k < K_mat; ++k) for (int n = 0; n < OUT_COLS; ++n) S_b = std::max(S_b, std::fabs(B_real[k][n]));

    Systolic sys;
    sys.reset_window_state();

    for (int r = 0; r < OUT_ROWS; ++r) {
        std::array<mag_t, K> mag{}; std::array<bool, K> sign{};
        for (int d = 0; d < K_mat; ++d) quantize(A_real[r][d], S_a, mag[d], sign[d]);
        sys.load_inputs(r, mag, sign);
    }
    for (int c = 0; c < OUT_COLS; ++c) {
        std::array<mag_t, K> mag{}; std::array<bool, K> sign{};
        for (int d = 0; d < K_mat; ++d) quantize(B_real[d][c], S_b, mag[d], sign[d]);
        sys.load_weights(c, mag, sign);
    }
    for (int t = 0; t < T / M; ++t) sys.tick();  // T/M ticks per window (M samples per tick)
    int drain[OUT_ROWS][OUT_COLS] = {};
    // Drain order: east-most tile first (tile_col counts down), and within a
    // tile, its N_W PE-columns forward (j counts up) -- see
    // Tile::shift_drain / Systolic::tick's drain phase.
    for (int dr = 0; dr < P_COLS * N_W; ++dr) {
        sys.tick();
        int tile_col = P_COLS - 1 - dr / N_W;
        int j        = dr % N_W;
        int pc       = tile_col * N_W + j;      // physical (global) column
        if (pc < OUT_COLS) for (int r = 0; r < OUT_ROWS; ++r) drain[r][pc] = sys.east_row(r);
    }

    // Dequantize (scmp's net per-term scale): real A*B ~= drain * S_a * S_b / T,
    // with S = abs_max / q_max (= abs_max / MAG_MAX).
    double dequant = S_a * S_b / double(T);

    double O_sc[OUT_ROWS][OUT_COLS], O_true[OUT_ROWS][OUT_COLS];
    double se = 0, st = 0, maxe = 0;
    for (int r = 0; r < OUT_ROWS; ++r)
        for (int c = 0; c < OUT_COLS; ++c) {
            O_sc[r][c] = drain[r][c] * dequant;
            double s = 0;
            for (int k = 0; k < K_mat; ++k) s += A_real[r][k] * B_real[k][c];
            O_true[r][c] = s;
            double e = O_sc[r][c] - O_true[r][c];
            se += e * e; st += O_true[r][c] * O_true[r][c];
            maxe = std::max(maxe, std::fabs(e));
        }
    double rms_err  = std::sqrt(se / (OUT_ROWS * OUT_COLS));
    double rms_true = std::sqrt(st / (OUT_ROWS * OUT_COLS));
    double rel      = 100.0 * rms_err / rms_true;

    // Greppable one-liner (scale-invariant accuracy metric).
    std::cout << "PRECISION  MAG_BITS=" << MAG_BITS << "  T=" << T
              << "  rms_rel_err=" << rel << "%"
              << "  (rms_err=" << rms_err << ", max_err=" << maxe << ")\n";

    std::cout << "O_true (FP reference A*B), " << OUT_ROWS << " x " << OUT_COLS << ":\n";
    for (int r = 0; r < OUT_ROWS; ++r) {
        std::cout << "  ";
        for (int c = 0; c < OUT_COLS; ++c) std::cout << O_true[r][c] << '\t';
        std::cout << '\n';
    }
    std::cout << "O_sc (SC array, dequantized), " << OUT_ROWS << " x " << OUT_COLS << ":\n";
    for (int r = 0; r < OUT_ROWS; ++r) {
        std::cout << "  ";
        for (int c = 0; c < OUT_COLS; ++c) std::cout << O_sc[r][c] << '\t';
        std::cout << '\n';
    }

    Stats s = sys.total_stats();
    std::cout << "Activity: and_ones=" << s.and_ones
              << " link_toggle=" << s.link_toggle
              << " sign_link_toggle=" << s.sign_link_toggle
              << " cmp_eval=" << s.cmp_eval
              << " rng_advance=" << s.rng_advance << '\n';
}

// ---------------------------------------------------------------------------
// Workload mode: read quantized tile data from stdin, run the simulator,
// print per-chunk drain matrices to stdout.
//
// stdin format:
//   N_TILES=<n> N_CHUNKS=<c> A_ROWS=<r> W_COLS=<w> K=<k>
//   Then N_TILES * N_CHUNKS blocks, each block (one hardware window):
//     A_ROWS lines of K space-separated uint32  (A boundaries)
//     A_ROWS lines of K space-separated 0/1     (A signs)
//     W_COLS lines of K space-separated uint32  (W boundaries)
//     W_COLS lines of K space-separated 0/1     (W signs)
//
// stdout format:
//   RESULT N_TILES=<n> N_CHUNKS=<c> A_ROWS=<r> W_COLS=<w>
//   Then N_TILES * N_CHUNKS drain matrices:
//     A_ROWS lines of W_COLS space-separated int32 drain values
//   (Python accumulates and dequantizes; C++ only does the SC simulation.)
// ---------------------------------------------------------------------------
// Binary workload format (used when --binary-file is passed):
//
//   Header (ASCII, one line):
//     "N_TILES=<n> N_CHUNKS=<c> A_ROWS=<r> W_COLS=<w> K=<k> K_BLOCKS=<kb>\n"
//   K_BLOCKS defaults to 1 if omitted (old workload files still work
//   unchanged).
//
//   Data (raw binary, immediately after the newline): for each tile
//   (N_TILES), for each chunk (N_CHUNKS), for each K-block (K_BLOCKS):
//       uint16_t  A_bnd[A_ROWS][K]   — magnitude boundaries (0..RNG_LEVELS-1)
//       uint8_t   A_sgn[A_ROWS][K]   — sign bits (0=pos, 1=neg)
//       uint16_t  W_bnd[W_COLS][K]
//       uint8_t   W_sgn[W_COLS][K]
//   (Little-endian; boundary fits in uint16 for MAG_BITS<=15, uint32 otherwise.
//    Python packs with np.uint16 so both sides must agree on the width.)
//
//   One CHUNK is one independent accumulate window (fresh RNG reset/reseed),
//   spanning K_BLOCKS*K of the GEMM's real reduction depth. Its K_BLOCKS are
//   accumulated together into a single drain value -- the RNG does NOT reset
//   between K-blocks within a chunk, matching designs/sc_gemm_3d/
//   inner_tile_2bit.sv's `acc` register, which only clears on clr_acc/reset,
//   never just from another cycle running. Each K-block after the first
//   burns 2 throwaway RNG steps per lane before its accumulate cycles run,
//   mirroring the RTL's 2-cycle pipeline-fill latency after an operand
//   reload (see tb_gemm.sv's run_spatial_tile / sc_kernel.py's
//   compute_spatial_tile, the RTL/numpy reference this mirrors).
//
//   Drain output (binary, written to stdout, after one ASCII header line):
//     Header: "RESULT N_TILES=<n> N_CHUNKS=<c> A_ROWS=<r> W_COLS=<w> CYCLES=1\n"
//     Data:   int32_t  drain [N_TILES][N_CHUNKS][A_ROWS][W_COLS]
//             uint64_t cycles[N_TILES][N_CHUNKS]   -- real hardware cycles for
//               that chunk's whole window: T/M ticks per K-block, +2 pipeline-
//               fill cycles for each K-block after the first (see the
//               K_BLOCKS note above), + P_COLS*N_W ticks for the drain-shift-
//               out phase (this function reads drain_reg directly instead of
//               physically running Tile::shift_drain -- see bipolar_matmul_demo
//               for the code path that actually ticks the drain phase -- so
//               those cycles are added in analytically rather than incurred).
//     (One drain value AND one cycle count per chunk, not per K-block.)
//     CYCLES=1 lets a reader detect whether the trailing cycles array is
//     present (older parsers that only read the header's N_TILES/N_CHUNKS/
//     A_ROWS/W_COLS prefix and the drain array are unaffected either way).

static void parse_header(const std::string& line,
                         int& n_tiles, int& n_chunks,
                         int& arows,   int& wcols, int& k, int& k_blocks) {
    size_t pos = 0;
    while (pos < line.size()) {
        size_t eq = line.find('=', pos);
        if (eq == std::string::npos) break;
        std::string key = line.substr(pos, eq - pos);
        size_t end = line.find(' ', eq);
        if (end == std::string::npos) end = line.size();
        int val = std::stoi(line.substr(eq + 1, end - eq - 1));
        if      (key == "N_TILES")  n_tiles  = val;
        else if (key == "N_CHUNKS") n_chunks = val;
        else if (key == "A_ROWS")   arows    = val;
        else if (key == "W_COLS")   wcols    = val;
        else if (key == "K")        k        = val;
        else if (key == "K_BLOCKS") k_blocks = val;
        pos = (end == line.size()) ? end : end + 1;
    }
}

static void run_workload_binary(const std::string& input_path) {
    constexpr int AROWS = P_ROWS * N_H;
    constexpr int WCOLS = P_COLS * N_W;
    // Bytes per (tile, chunk, K-block) element in the packed binary stream:
    //   uint16 A_bnd[AROWS][K]  +  uint8 A_sgn[AROWS][K]
    //   uint16 W_bnd[WCOLS][K]  +  uint8 W_sgn[WCOLS][K]
    constexpr size_t KBLOCK_BYTES =
        size_t(AROWS) * K * 2 + size_t(AROWS) * K +
        size_t(WCOLS) * K * 2 + size_t(WCOLS) * K;

    FILE* fin = std::fopen(input_path.c_str(), "rb");
    if (!fin) { std::perror(input_path.c_str()); std::exit(1); }

    // Read ASCII header line
    char hdr[512] = {};
    if (!std::fgets(hdr, sizeof(hdr), fin)) { std::fputs("empty file\n", stderr); std::exit(1); }

    int n_tiles = 0, n_chunks = 0, arows = 0, wcols = 0, k = 0, k_blocks = 1;
    parse_header(std::string(hdr), n_tiles, n_chunks, arows, wcols, k, k_blocks);

    if (arows != AROWS || wcols != WCOLS || k != K) {
        std::fprintf(stderr,
            "workload mismatch: file has A_ROWS=%d W_COLS=%d K=%d but "
            "hardware compiled with A_ROWS=%d W_COLS=%d K=%d\n",
            arows, wcols, k, AROWS, WCOLS, K);
        std::exit(1);
    }

    const size_t ELEM_BYTES = KBLOCK_BYTES * size_t(k_blocks);

    // Pre-read all binary data into memory so tiles can be processed in parallel.
    size_t n_elem = size_t(n_tiles) * n_chunks;
    std::vector<uint8_t> raw(n_elem * ELEM_BYTES);
    if (std::fread(raw.data(), 1, raw.size(), fin) != raw.size()) {
        std::fputs("unexpected EOF reading tile data\n", stderr);
        std::exit(1);
    }
    std::fclose(fin);

    // Emit drain output header before the parallel section
    std::printf("RESULT N_TILES=%d N_CHUNKS=%d A_ROWS=%d W_COLS=%d CYCLES=1\n",
                n_tiles, n_chunks, AROWS, WCOLS);
    std::fflush(stdout);

    std::vector<int32_t>  drain_out(n_elem * AROWS * WCOLS, 0);
    std::vector<uint64_t> cycles_out(n_elem, 0);

    // One Systolic object per thread (each is stateful: RNG, PE registers).
    int max_threads = omp_get_max_threads();
    std::vector<Systolic> sys_pool(max_threads);

    #pragma omp parallel for schedule(static)
    for (int tile = 0; tile < n_tiles; ++tile) {
        Systolic& sys = sys_pool[omp_get_thread_num()];

        for (int ch = 0; ch < n_chunks; ++ch) {
            const uint8_t* chunk_base = raw.data() + (size_t(tile) * n_chunks + ch) * ELEM_BYTES;

            sys.reset_window_state();
            sys.cycle      = 0;   // per-chunk cycle count, not a running total across chunks
            sys.rng_input  = make_sobol_bank(SEED_BASE_IN, SEED_STRIDE_IN, SOBOL_V1.data());
            sys.rng_weight = make_sobol_bank(SEED_BASE_W, SEED_STRIDE_W, SOBOL_V2.data());

            for (int kb = 0; kb < k_blocks; ++kb) {
                const uint8_t* p = chunk_base + size_t(kb) * KBLOCK_BYTES;
                const uint16_t* A_bnd = reinterpret_cast<const uint16_t*>(p);
                const uint8_t*  A_sgn = p + AROWS * K * 2;
                const uint16_t* W_bnd = reinterpret_cast<const uint16_t*>(A_sgn + AROWS * K);
                const uint8_t*  W_sgn = reinterpret_cast<const uint8_t*>(W_bnd) + WCOLS * K * 2;

                std::array<mag_t, K> mag; std::array<bool, K> sgn;
                for (int r = 0; r < AROWS; ++r) {
                    for (int d = 0; d < K; ++d) {
                        mag[d] = mag_t(A_bnd[r * K + d]);
                        sgn[d] = bool(A_sgn[r * K + d]);
                    }
                    sys.load_inputs(r, mag, sgn);
                }
                for (int c = 0; c < WCOLS; ++c) {
                    for (int d = 0; d < K; ++d) {
                        mag[d] = mag_t(W_bnd[c * K + d]);
                        sgn[d] = bool(W_sgn[c * K + d]);
                    }
                    sys.load_weights(c, mag, sgn);
                }

                // Pipeline-fill skip between K-blocks (not before the first):
                // discarded comparator output during these cycles never
                // reaches an accumulator either way, so only the RNG state
                // advance matters for bit-exactness -- see the format doc
                // comment above.
                if (kb > 0) {
                    for (int f = 0; f < 2; ++f)
                        for (int m = 0; m < M; ++m) {
                            sys.rng_input[m].step();
                            sys.rng_weight[m].step();
                        }
                    sys.cycle += 2;   // the 2-cycle pipeline-fill latency itself
                }

                for (int t = 0; t < T / M; ++t) sys.step_compute();
            }
            sys.end_window(k_blocks);
            // Drain-shift-out phase: this loop reads drain_reg directly rather
            // than physically running Tile::shift_drain (see the format doc
            // comment above), so add its P_COLS*N_W ticks in analytically.
            sys.cycle += static_cast<uint64_t>(P_COLS) * N_W;

            int idx = tile * n_chunks + ch;
            cycles_out[idx] = sys.cycle;
            int base = idx * AROWS * WCOLS;
            for (int r = 0; r < AROWS; ++r) {
                int tr = r / N_H, ti = r % N_H;
                for (int c = 0; c < WCOLS; ++c) {
                    int tc = c / N_W, tj = c % N_W;
                    drain_out[base + r * WCOLS + c] =
                        int32_t(sys.tiles[tr][tc].drain_reg[ti][tj]);
                }
            }
        }
    }

    std::fwrite(drain_out.data(), sizeof(int32_t), drain_out.size(), stdout);
    std::fwrite(cycles_out.data(), sizeof(uint64_t), cycles_out.size(), stdout);
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--info") {
            // Machine-readable hardware config for the Python bridge.
            std::cout << "P_ROWS=" << P_ROWS << " P_COLS=" << P_COLS
                      << " N_H=" << N_H << " N_W=" << N_W
                      << " K=" << K << " M=" << M
                      << " T=" << T << " MAG_BITS=" << MAG_BITS
                      << " A_ROWS=" << P_ROWS*N_H << " W_COLS=" << P_COLS*N_W << "\n";
            return 0;
        }
        if (arg == "--binary-file" && i + 1 < argc) {
            run_workload_binary(std::string(argv[++i]));
            return 0;
        }
    }
    std::cout << "Architecture (sign-magnitude bipolar SC): K=" << K
              << ", N_H=" << N_H << ", N_W=" << N_W
              << ", P_ROWS=" << P_ROWS << ", P_COLS=" << P_COLS
              << ", M=" << M
              << ", T=" << T << ", MAG_BITS=" << MAG_BITS
              << ", MAG_MAX=" << MAG_MAX << '\n';
    bipolar_matmul_demo();
    return 0;
}
