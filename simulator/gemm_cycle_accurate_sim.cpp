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
//   * Each AND-1 event is routed by its lane sign into popcnt_pos or
//     popcnt_neg, accumulated into oreg_pos / oreg_neg respectively.
//   * drain_reg = oreg_pos - oreg_neg  (no bias; sc_matmul is A*B only).
//
// Systolic geometry (N_ROWS x N_COLS tiles, may be non-square)
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
//   for the first (N_ROWS-1-r) cycles (the links start at 0).  This is real
//   hardware latency; we keep it.  It costs <= (array side - 1) of T cycles,
//   so it matters more at short T -- run extra pipeline-fill cycles if needed.
//
// Decode / dequant (matches scmp's net per-term scale)
//   A lane's expected AND count over T cycles is
//     T * (boundary_a / RNG_LEVELS) * (boundary_b / RNG_LEVELS)
//       ~= T * (mag_a / q_max) * (mag_w / q_max),
//   so  drain = oreg_pos - oreg_neg ~= (sum mag_a*mag_w) * T / q_max^2.
//   With per-tensor scale S = abs_max / q_max, the real dot product is
//     real = drain * S_a * S_b / T
//   which is exactly scmp's decode count*(q_max^2/stoc_len)*(S_a)*(S_b)
//   after substituting S = abs_max/q_max.  No SC_DENOM, no bias.
//
// Lane decorrelation (CFG_OWEN, default off)
//   All K lanes on one side share ONE rng threshold per cycle (the cheap
//   "amortized RNG" choice), so the lanes are correlated and the dot product
//   forfeits the 1/sqrt(K) error averaging it would get from independent
//   per-lane randomness.  CFG_OWEN=1 XORs each lane's copy of the shared
//   threshold with a distinct stationary mask (a Weyl/golden-ratio constant)
//   -- a per-lane wire tap, no extra RNG -- which partially restores that
//   averaging.  See make_owen() and OWEN_IN / OWEN_W.
//
// Energy accounting -- what is and isn't counted (see struct Stats):
//   counted    : AND-1 events, accumulator updates, mag wire toggles, the
//                once-per-window sign wire toggles, drain shifts, edge loads,
//                RNG steps, comparator evals.
//   NOT counted: AND-0 outputs (no switching), the stationary sign XNOR (no
//                per-cycle switching), popcount-tree internal adds (only the
//                net accumulator increment is priced).
//
// Build: `make` (see Makefile), or directly
//   g++ -std=c++17 -O2 -Wall -Wextra tile.cpp -o tile_sim
// Override knobs at compile time, e.g.:  make CFG="-DCFG_K=8 -DCFG_OWEN=1"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>

// ---- Tunable parameters --------------------------------------------------
// Every knob has a default below and can be overridden at COMPILE time, e.g.:
//   g++ -std=c++17 -O2 -DCFG_MAG_BITS=8 -DCFG_K=8 -DCFG_N_ROWS=2 -DCFG_N_COLS=6 tile.cpp
// The CFG_* macros are just the override hooks; the typed constexpr constants
// below are what the rest of the code uses.
#ifndef CFG_MAG_BITS
#define CFG_MAG_BITS 7                  // magnitude bit width (precision)
#endif
#ifndef CFG_K
#define CFG_K 4                         // diagonal length per tile (dot-product depth)
#endif
#ifndef CFG_N_ROWS
#define CFG_N_ROWS 4                    // systolic tile rows (max output rows / window)
#endif
#ifndef CFG_N_COLS
#define CFG_N_COLS 4                    // systolic tile cols (max output cols / window)
#endif
#ifndef CFG_STOC_LEN
#define CFG_STOC_LEN (1 << (CFG_MAG_BITS + 1)) // bitstream length T; default 2^sc_prec (scmp default)
#endif
#ifndef CFG_OWEN
#define CFG_OWEN 0                       // 1 = per-lane Owen XOR scramble (decorrelate K lanes)
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
constexpr int   N_ROWS     = CFG_N_ROWS;                    // systolic tile rows
constexpr int   N_COLS     = CFG_N_COLS;                    // systolic tile cols

// Activity counters used as proxies for dynamic energy.
struct Stats {
    uint64_t and_ones         = 0;  // AND outputs that were 1 (AND-array switching)
    uint64_t and_pos          = 0;  // AND-1 events routed to popcnt_pos
    uint64_t and_neg          = 0;  // AND-1 events routed to popcnt_neg
    uint64_t oreg_inc         = 0;  // non-zero accumulator updates (either polarity)
    uint64_t add_op           = 0;  // adds  (pos accum, neg accum)
    uint64_t sub_op           = 0;  // subs  (oreg_pos - oreg_neg at drain)
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

// Per-lane Owen XOR masks (only active when CFG_OWEN=1).  The K lanes of a
// tile share ONE rng threshold; XOR-ing each lane's threshold with a distinct
// stationary mask decorrelates the lanes (a bijection on [0,RNG_LEVELS), so
// each lane's marginal -- and thus its encoded magnitude -- is unchanged).
// Masks spread the K lanes evenly across the threshold domain; weights use a
// half-step offset so input and weight lane-masks differ.  This is a wire tap,
// no extra RNG -- the same trick as scmp_kernels' _owen_scramble.
constexpr std::array<mag_t, K> make_owen(mag_t salt) {
    std::array<mag_t, K> m{};
    // Weyl / golden-ratio multiplier: odd (coprime to the RNG_LEVELS modulus),
    // ~0.618 * RNG_LEVELS.  d*GOLDEN spreads the lane index across ALL threshold
    // bits, not just the high ones -> far stronger lane decorrelation than a
    // plain high-bit step, and it keeps working as K grows.
    constexpr mag_t GOLDEN =
        mag_t(0.6180339887498949 * double(RNG_LEVELS)) | mag_t(1);
    for (int d = 0; d < K; ++d)
        m[d] = CFG_OWEN ? ((mag_t(d) * GOLDEN + salt) & RNG_MASK) : mag_t(0);
    return m;
}
constexpr std::array<mag_t, K> OWEN_IN = make_owen(0);
constexpr std::array<mag_t, K> OWEN_W  = make_owen(RNG_LEVELS >> 1);

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

// Tile: K diagonal magnitude ANDs + K stationary-sign XNORs feeding split
// popcounts into oreg_pos / oreg_neg.  Output = oreg_pos - oreg_neg (A*B).
class Tile {
public:
    bool is_input_edge  = false;   // west column: comparator drives in_mag_bits from in_mag
    bool is_weight_edge = false;   // south row : comparator drives w_mag_bits  from w_mag

    // Stationary operand registers (held for a whole window).
    // in_mag/w_mag hold the remapped boundary (0..RNG_LEVELS), not raw magnitude.
    std::array<mag_t, K> in_mag{},  w_mag{};
    std::array<bool,  K> in_sign{}, w_sign{};  // 1-bit signs (true = negative)

    // Per-cycle stream bits -- MAGNITUDE only (signs are stationary, above).
    std::array<bool, K> in_mag_bits{}, w_mag_bits{};
    // Pass-through stream bits latched into the downstream inter-tile flops.
    std::array<bool, K> in_mag_out{},  w_mag_out{};

    int oreg_pos  = 0;
    int oreg_neg  = 0;
    int drain_reg = 0;
    Stats stats;

    // One combinational cycle.  Edge faces generate this cycle's stream bit
    // from the comparator; non-edge faces were already driven by Systolic from
    // the inter-tile flop latched LAST cycle (true systolic, one hop/cycle).
    void compute(mag_t rng_i, mag_t rng_w) {
        if (is_input_edge) {
            for (int d = 0; d < K; ++d) {
                mag_t thr_i = (rng_i ^ OWEN_IN[d]) & RNG_MASK;   // per-lane scramble (no-op if CFG_OWEN=0)
                in_mag_bits[d] = in_mag[d] > thr_i;              // in_mag holds the remapped boundary
                ++stats.cmp_eval;
            }
        }
        if (is_weight_edge) {
            for (int d = 0; d < K; ++d) {
                mag_t thr_w = (rng_w ^ OWEN_W[d]) & RNG_MASK;
                w_mag_bits[d] = w_mag[d] > thr_w;
                ++stats.cmp_eval;
            }
        }
        // K diagonal magnitude ANDs; lane sign = in_sign XNOR w_sign (read
        // from the stationary registers, no per-cycle switching).
        int popcnt_pos = 0, popcnt_neg = 0;
        for (int d = 0; d < K; ++d) {
            bool a = in_mag_bits[d] && w_mag_bits[d];
            if (a) {
                bool neg = (in_sign[d] != w_sign[d]);   // signs differ => product negative
                if (neg) { ++popcnt_neg; ++stats.and_neg; }
                else     { ++popcnt_pos; ++stats.and_pos; }
                ++stats.and_ones;
            }
            in_mag_out[d] = in_mag_bits[d];
            w_mag_out[d]  = w_mag_bits[d];
        }
        if (popcnt_pos) { oreg_pos += popcnt_pos; ++stats.oreg_inc; ++stats.add_op; }
        if (popcnt_neg) { oreg_neg += popcnt_neg; ++stats.oreg_inc; ++stats.add_op; }
    }

    // Drain combine -- pos/neg meet here. (No bias; sc_matmul computes A*B only.)
    void load_drain() {
        drain_reg = oreg_pos - oreg_neg;
        ++stats.sub_op;   // oreg_pos - oreg_neg
        oreg_pos = 0;
        oreg_neg = 0;
    }

    void shift_drain(int from_west) {
        if (drain_reg != from_west) ++stats.drain_shift;
        drain_reg = from_west;
    }

    // Clear transient per-window state. Stationary operands and Stats kept.
    void reset_window() {
        oreg_pos = 0; oreg_neg = 0; drain_reg = 0;
        in_mag_bits.fill(false); w_mag_bits.fill(false);
        in_mag_out.fill(false);  w_mag_out.fill(false);
    }
};

// Systolic: owns the two array-wide Sobols, the inter-tile flops (mag + sign
// halves, each 1 bit per lane), and the tile grid.
//   tiles[0][.]          top row
//   tiles[N_ROWS-1][.]   south (weight) edge
//   tiles[.][0]          west (input) edge
//   tiles[N_ROWS-1][0]   corner (does both conversions)
class Systolic {
public:
    Sobol rng_input, rng_weight;
    std::array<std::array<Tile, N_COLS>, N_ROWS> tiles;
    // MAGNITUDE bitstream flops (toggle every cycle).  link_*[r][c] feeds tile (r,c).
    std::array<std::array<std::array<bool, K>, N_COLS>, N_ROWS> link_we_mag{};
    std::array<std::array<std::array<bool, K>, N_COLS>, N_ROWS> link_ns_mag{};
    // SIGN flops (1 bit per lane).  Written once per window by load_*; held all T cycles.
    std::array<std::array<std::array<bool, K>, N_COLS>, N_ROWS> link_we_sign{};
    std::array<std::array<std::array<bool, K>, N_COLS>, N_ROWS> link_ns_sign{};
    std::array<int, N_ROWS> east_out{};
    Stats systolic_stats;
    int  tick_in_phase = 0;
    bool computing     = true;
    uint64_t cycle     = 0;

    Systolic(const mag_t* dirvec_i = SOBOL_V1.data(),
             const mag_t* dirvec_w = SOBOL_V2.data())
        : rng_input(dirvec_i), rng_weight(dirvec_w) {
        for (int r = 0; r < N_ROWS; ++r) tiles[r][0].is_input_edge         = true;
        for (int c = 0; c < N_COLS; ++c) tiles[N_ROWS-1][c].is_weight_edge = true;
    }

    // Load A's row r.  Magnitude is written only to the west-edge tile -- it
    // becomes the bitstream via the comparator and marches east one tile per
    // cycle through link_we_mag.  Sign is constant for the whole stream, so it
    // is set into each tile's sign register and the sign flops; those flops
    // change at most once per window (sign_link_toggle counts only real changes).
    void load_inputs(int r, const std::array<mag_t, K>& mag,
                            const std::array<bool,  K>& sign) {
        tiles[r][0].in_mag  = mag;
        tiles[r][0].in_sign = sign;
        tiles[r][0].stats.bin_load += 2 * K;   // edge mag reg + edge sign reg
        for (int c = 1; c < N_COLS; ++c) {
            for (int d = 0; d < K; ++d)
                if (link_we_sign[r][c][d] != sign[d]) ++systolic_stats.sign_link_toggle;
            link_we_sign[r][c]  = sign;         // one-shot per-window sign hop
            tiles[r][c].in_sign = sign;         // tile's stationary sign register
        }
    }
    // Load B's column c (mirror of the above, weights flow south->north).
    void load_weights(int c, const std::array<mag_t, K>& mag,
                             const std::array<bool,  K>& sign) {
        tiles[N_ROWS-1][c].w_mag  = mag;
        tiles[N_ROWS-1][c].w_sign = sign;
        tiles[N_ROWS-1][c].stats.bin_load += 2 * K;
        for (int r = 0; r < N_ROWS - 1; ++r) {
            for (int d = 0; d < K; ++d)
                if (link_ns_sign[r][c][d] != sign[d]) ++systolic_stats.sign_link_toggle;
            link_ns_sign[r][c]  = sign;
            tiles[r][c].w_sign  = sign;
        }
    }

    // Clear inter-tile MAG flops and per-tile transient bits (NOT sign flops,
    // NOT oreg/drain -- signs are reloaded each window, oreg handled by drain).
    void clear_window_transient() {
        for (auto& row : link_we_mag) for (auto& cell : row) cell.fill(false);
        for (auto& row : link_ns_mag) for (auto& cell : row) cell.fill(false);
        for (auto& row : tiles)
            for (auto& t : row) {
                t.in_mag_bits.fill(false); t.w_mag_bits.fill(false);
                t.in_mag_out.fill(false);  t.w_mag_out.fill(false);
            }
    }

    void reset_window_state() {
        clear_window_transient();
        for (auto& row : tiles) for (auto& t : row) { t.oreg_pos = 0; t.oreg_neg = 0; t.drain_reg = 0; }
        east_out.fill(0);
        tick_in_phase = 0;
        computing     = true;
    }

    void tick() {
        ++cycle;
        if (computing) {
            // (1) advance both RNGs (one shared threshold per operand, array-wide)
            mag_t ri = rng_input.step();
            mag_t rw = rng_weight.step();
            systolic_stats.rng_advance += 2;

            // (2) drive non-edge faces from the inter-tile flop latched LAST
            //     cycle -- this IS the systolic hop (one tile per cycle).
            //     Edge faces are generated inside compute() from the comparator.
            for (int r = 0; r < N_ROWS; ++r)
                for (int c = 0; c < N_COLS; ++c) {
                    if (!tiles[r][c].is_input_edge)  tiles[r][c].in_mag_bits = link_we_mag[r][c];
                    if (!tiles[r][c].is_weight_edge) tiles[r][c].w_mag_bits  = link_ns_mag[r][c];
                }

            // (3) every tile computes combinationally (reads its own bits only)
            for (int r = 0; r < N_ROWS; ++r)
                for (int c = 0; c < N_COLS; ++c)
                    tiles[r][c].compute(ri, rw);

            // (4) latch this cycle's pass-through MAG bits into the downstream
            //     flops (read next cycle).  Signs are stationary -> untouched.
            for (int r = 0; r < N_ROWS; ++r)
                for (int c = 0; c < N_COLS; ++c) {
                    if (c + 1 < N_COLS) {                 // input bit hops east
                        for (int d = 0; d < K; ++d)
                            if (link_we_mag[r][c+1][d] != tiles[r][c].in_mag_out[d])
                                ++systolic_stats.link_toggle;
                        link_we_mag[r][c+1] = tiles[r][c].in_mag_out;
                    }
                    if (r > 0) {                          // weight bit hops north
                        for (int d = 0; d < K; ++d)
                            if (link_ns_mag[r-1][c][d] != tiles[r][c].w_mag_out[d])
                                ++systolic_stats.link_toggle;
                        link_ns_mag[r-1][c] = tiles[r][c].w_mag_out;
                    }
                }

            // (5) end of window -> drain latch, switch phase.
            if (++tick_in_phase == T) {
                for (auto& row : tiles) for (auto& t : row) t.load_drain();
                clear_window_transient();
                computing     = false;
                tick_in_phase = 0;
            }
        } else {
            // Drain: capture east column, then shift right-to-left.
            for (int r = 0; r < N_ROWS; ++r) east_out[r] = tiles[r][N_COLS-1].drain_reg;
            for (int r = 0; r < N_ROWS; ++r) {
                for (int c = N_COLS - 1; c > 0; --c)
                    tiles[r][c].shift_drain(tiles[r][c-1].drain_reg);
                tiles[r][0].drain_reg = 0;
            }
            if (++tick_in_phase == N_COLS) { computing = true; tick_in_phase = 0; }
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
    constexpr int M_mat = N_ROWS, K_mat = K, N_mat = N_COLS;

    // Fixed real operands (do NOT depend on MAG_BITS). No bias: O = A*B.
    double A_real[M_mat][K_mat], B_real[K_mat][N_mat];
    for (int m = 0; m < M_mat; ++m)
        for (int k = 0; k < K_mat; ++k)
            A_real[m][k] = 1.3 * std::sin(0.9 * (m * K_mat + k) + 0.3);
    for (int k = 0; k < K_mat; ++k)
        for (int n = 0; n < N_mat; ++n)
            B_real[k][n] = 0.8 * std::cos(0.7 * (k * N_mat + n) + 0.4);

    // Per-tensor abs-max scales so operands normalize into [-1, 1].
    double S_a = 1e-9, S_b = 1e-9;
    for (int m = 0; m < M_mat; ++m) for (int k = 0; k < K_mat; ++k) S_a = std::max(S_a, std::fabs(A_real[m][k]));
    for (int k = 0; k < K_mat; ++k) for (int n = 0; n < N_mat; ++n) S_b = std::max(S_b, std::fabs(B_real[k][n]));

    Systolic sys;
    sys.reset_window_state();

    for (int r = 0; r < M_mat; ++r) {
        std::array<mag_t, K> mag{}; std::array<bool, K> sign{};
        for (int d = 0; d < K_mat; ++d) quantize(A_real[r][d], S_a, mag[d], sign[d]);
        sys.load_inputs(r, mag, sign);
    }
    for (int c = 0; c < N_mat; ++c) {
        std::array<mag_t, K> mag{}; std::array<bool, K> sign{};
        for (int d = 0; d < K_mat; ++d) quantize(B_real[d][c], S_b, mag[d], sign[d]);
        sys.load_weights(c, mag, sign);
    }
    for (int t = 0; t < T; ++t) sys.tick();
    int drain[M_mat][N_mat] = {};
    for (int dr = 0; dr < N_COLS; ++dr) {
        sys.tick();
        int tc = N_COLS - 1 - dr;          // east-first drain order
        if (tc < N_mat) for (int r = 0; r < M_mat; ++r) drain[r][tc] = sys.east_row(r);
    }

    // Dequantize (scmp's net per-term scale): real A*B ~= drain * S_a * S_b / T,
    // with S = abs_max / q_max (= abs_max / MAG_MAX).
    double dequant = S_a * S_b / double(T);

    double O_sc[M_mat][N_mat], O_true[M_mat][N_mat];
    double se = 0, st = 0, maxe = 0;
    for (int r = 0; r < M_mat; ++r)
        for (int c = 0; c < N_mat; ++c) {
            O_sc[r][c] = drain[r][c] * dequant;
            double s = 0;
            for (int k = 0; k < K_mat; ++k) s += A_real[r][k] * B_real[k][c];
            O_true[r][c] = s;
            double e = O_sc[r][c] - O_true[r][c];
            se += e * e; st += O_true[r][c] * O_true[r][c];
            maxe = std::max(maxe, std::fabs(e));
        }
    double rms_err  = std::sqrt(se / (M_mat * N_mat));
    double rms_true = std::sqrt(st / (M_mat * N_mat));
    double rel      = 100.0 * rms_err / rms_true;

    // Greppable one-liner (scale-invariant accuracy metric).
    std::cout << "PRECISION  MAG_BITS=" << MAG_BITS << "  T=" << T
              << "  rms_rel_err=" << rel << "%"
              << "  (rms_err=" << rms_err << ", max_err=" << maxe << ")\n";

    std::cout << "O_true (FP reference A*B), " << M_mat << " x " << N_mat << ":\n";
    for (int r = 0; r < M_mat; ++r) {
        std::cout << "  ";
        for (int c = 0; c < N_mat; ++c) std::cout << O_true[r][c] << '\t';
        std::cout << '\n';
    }
    std::cout << "O_sc (SC array, dequantized), " << M_mat << " x " << N_mat << ":\n";
    for (int r = 0; r < M_mat; ++r) {
        std::cout << "  ";
        for (int c = 0; c < N_mat; ++c) std::cout << O_sc[r][c] << '\t';
        std::cout << '\n';
    }

    Stats s = sys.total_stats();
    std::cout << "Activity: and_ones=" << s.and_ones
              << " link_toggle=" << s.link_toggle
              << " sign_link_toggle=" << s.sign_link_toggle
              << " cmp_eval=" << s.cmp_eval
              << " rng_advance=" << s.rng_advance << '\n';
}

int main() {
    std::cout << "Architecture (sign-magnitude bipolar SC): K=" << K
              << ", N_ROWS=" << N_ROWS << ", N_COLS=" << N_COLS
              << ", T=" << T << ", MAG_BITS=" << MAG_BITS
              << ", MAG_MAX=" << MAG_MAX << '\n';
    bipolar_matmul_demo();
    return 0;
}
