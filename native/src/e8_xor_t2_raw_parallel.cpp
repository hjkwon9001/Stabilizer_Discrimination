#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// Exact raw Bellman solver for k-copy E8 XOR/sign parity.
// sigma_tau = [I + (-1)^tau W^{tensor k}]/2^(3k), with equal priors.
// Every nonidentity first Pauli is evaluated for a complete normal run.
// Each worker owns a memo table, reset between fixed-outcome root branches.
// This implementation deliberately applies no unverified symmetry reductions.

#ifndef XOR_COPIES
#define XOR_COPIES 2
#endif
static_assert(XOR_COPIES >= 1 && XOR_COPIES <= 4,
              "E8 XOR supports XOR_COPIES in [1,4]");

namespace {

constexpr int COPIES = XOR_COPIES;
constexpr int N = 3 * COPIES;
constexpr int NB = 2 * N;
constexpr int UMAX = 1 << NB;
constexpr int SLOTBITS = NB + 1;
constexpr int KEY_WORDS = (N * SLOTBITS + 63) / 64;
constexpr int COEFF_DEN = 1 << (2 * COPIES);
constexpr int NUM_HYP = 2;
constexpr int VALUE_DEN = COEFF_DEN * NUM_HYP * (1 << N);
constexpr int DEFAULT_CAPACITY_POWER = COPIES == 1 ? 12 : (COPIES == 2 ? 25 : 24);
// At the supported maximum, labels occupy 24 unsigned bits and keys 300 bits.
// Even the loose bound 2^N*(COEFF_DEN+2^N) < 2^25 on an accumulated
// numerator fits int32_t. Exact comparisons use int64_t and __int128.
static_assert(NB < 31);
static_assert((int64_t{1} << N) * (COEFF_DEN + (int64_t{1} << N)) <
              (int64_t{1} << 31));

struct Val {
    int32_t a = 0;
    int32_t b = 0;
};  // (a + b sqrt(2)) / VALUE_DEN

inline Val addv(Val x, Val y) { return {x.a + y.a, x.b + y.b}; }
inline Val negv(Val x) { return {-x.a, -x.b}; }
inline bool eqv(Val x, Val y) { return x.a == y.a && x.b == y.b; }

int sign_exact(int64_t a, int64_t b) {
    if (b == 0) return (a > 0) - (a < 0);
    if (a == 0) return (b > 0) - (b < 0);
    if ((a > 0 && b > 0)) return 1;
    if ((a < 0 && b < 0)) return -1;
    const __int128 aa = static_cast<__int128>(a) * a;
    const __int128 bb = static_cast<__int128>(2) * b * b;
    if (aa == bb) return 0;
    if (a > 0 && b < 0) return aa > bb ? 1 : -1;
    return aa > bb ? -1 : 1;
}

inline bool lessv(Val x, Val y) {
    return sign_exact(static_cast<int64_t>(x.a) - y.a,
                      static_cast<int64_t>(x.b) - y.b) < 0;
}

inline double decv(Val x) {
    return (static_cast<double>(x.a) + static_cast<double>(x.b) * std::sqrt(2.0)) /
           VALUE_DEN;
}

std::string val_string(Val x) {
    std::ostringstream out;
    out << "(" << x.a << " + " << x.b << " sqrt(2))/" << VALUE_DEN;
    return out.str();
}

// Local labels I=0, X=1, Z=2, Y=3.  PH is the exponent e in AB=i^e C.
constexpr int OUT[4][4] = {
    {0, 1, 2, 3}, {1, 0, 3, 2}, {2, 3, 0, 1}, {3, 2, 1, 0}};
constexpr int PH[4][4] = {
    {0, 0, 0, 0}, {0, 0, 3, 1}, {0, 1, 0, 3}, {0, 3, 1, 0}};

inline int parity_u(unsigned x) { return std::popcount(x) & 1; }

inline int enc(int n, int x, int z, int sign = 0) {
    return x | (z << n) | ((sign & 1) << (2 * n));
}

inline void decp(int n, int p, int& x, int& z, int& sign) {
    sign = p >> (2 * n);
    const int u = p & ((1 << (2 * n)) - 1);
    x = u & ((1 << n) - 1);
    z = u >> n;
}

inline int mulp(int n, int p, int q) {
    int x, z, s, a, b, t;
    decp(n, p, x, z, s);
    decp(n, q, a, b, t);
    int xr = 0, zr = 0, phase = 0;
    for (int i = 0; i < n; ++i) {
        const int l = ((x >> i) & 1) | (((z >> i) & 1) << 1);
        const int r = ((a >> i) & 1) | (((b >> i) & 1) << 1);
        const int o = OUT[l][r];
        phase = (phase + PH[l][r]) & 3;
        xr |= (o & 1) << i;
        zr |= ((o >> 1) & 1) << i;
    }
    if (phase != 0 && phase != 2) {
        throw std::logic_error("Attempted to multiply anticommuting Paulis");
    }
    return enc(n, xr, zr, s ^ t ^ (phase >> 1));
}

std::vector<int> extend_code(const std::vector<int>& code, int q, int sign, int n) {
    const int sq = q | (sign << (2 * n));
    std::vector<int> out;
    out.reserve(code.size() * 2);
    out.insert(out.end(), code.begin(), code.end());
    for (int g : code) out.push_back(mulp(n, g, sq));
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<int> orth_basis3(int d) {
    std::vector<int> basis;
    std::vector<int> span{0};
    for (int b = 1; b < 8; ++b) {
        if (parity_u(static_cast<unsigned>(b & d)) != 0) continue;
        if (std::find(span.begin(), span.end(), b) != span.end()) continue;
        basis.push_back(b);
        const int size = static_cast<int>(span.size());
        for (int i = 0; i < size; ++i) span.push_back(span[i] ^ b);
        if (basis.size() == 2) return basis;
    }
    throw std::logic_error("Could not construct d-perp basis");
}

std::vector<int> state_code3(int u, int d, int sign) {
    std::vector<int> code{enc(3, 0, 0, 0)};
    code = extend_code(code, d, sign, 3);
    for (int b : orth_basis3(d)) {
        code = extend_code(code, b << 3, parity_u(static_cast<unsigned>(b & u)), 3);
    }
    return code;
}

// Factored Pauli coefficients avoid a dense 2*4^N hypothesis table.
std::array<int, 64> ONE_COPY_ETA{};
std::vector<std::pair<int, int>> ONE_COPY_SUPPORT;

Val hypothesis_coefficient(int hypothesis, int u) {
    if (u == 0) return {COEFF_DEN, 0};
    const int mask = (1 << N) - 1;
    const int x = u & mask;
    const int z = u >> N;
    int coefficient = hypothesis == 0 ? 1 : -1;
    for (int copy = 0; copy < COPIES; ++copy) {
        const int local = ((x >> (3 * copy)) & 7) |
                          (((z >> (3 * copy)) & 7) << 3);
        coefficient *= ONE_COPY_ETA[local];
        if (coefficient == 0) break;
    }
    return {coefficient, 0};
}

void build_hypotheses() {
    // W=(1/4) sum_p eta_p P_p, obtained from the four E8 plus states.
    // Its identity coefficient vanishes. For k copies, every tensor-support
    // coefficient is a product of k signs over 4^k.
    ONE_COPY_ETA.fill(0);
    ONE_COPY_SUPPORT.clear();
    // Match stabdisc.states: the first ket character is qubit zero.
    // The legacy two-copy program reversed the three local qubits.
    const int us[4] = {0b000, 0b010, 0b100, 0b001};
    const int ds[4] = {0b111, 0b001, 0b010, 0b100};
    for (int j = 0; j < 4; ++j) {
        for (int p : state_code3(us[j], ds[j], 0)) {
            int x, z, sg;
            decp(3, p, x, z, sg);
            ONE_COPY_ETA[x | (z << 3)] += sg ? -1 : 1;
        }
    }
    ONE_COPY_ETA[0] = 0;
    for (int u = 1; u < 64; ++u) {
        const int eta = ONE_COPY_ETA[u];
        if (eta == 0) continue;
        if (eta != 1 && eta != -1) {
            throw std::logic_error("Unexpected coefficient in one-copy W support");
        }
        ONE_COPY_SUPPORT.emplace_back(u, eta);
    }
    if (ONE_COPY_SUPPORT.size() != 16) {
        throw std::logic_error("One-copy W support must contain 16 Paulis");
    }
}

void dump_hypotheses() {
    build_hypotheses();
    std::cout << "# COPIES=" << COPIES << " COEFF_DEN=" << COEFF_DEN
              << " VALUE_DEN=" << VALUE_DEN << '\n';
    std::cout << "u,a0,b0,a1,b1\n0," << COEFF_DEN << ",0," << COEFF_DEN << ",0\n";
    std::vector<std::pair<int, int>> tensor{{0, 1}};
    for (int copy = 0; copy < COPIES; ++copy) {
        std::vector<std::pair<int, int>> next;
        next.reserve(tensor.size() * ONE_COPY_SUPPORT.size());
        for (const auto& [u, e] : tensor) {
            for (const auto& [local, eta] : ONE_COPY_SUPPORT) {
                const int shifted = ((local & 7) << (3 * copy)) |
                                    ((local >> 3) << (N + 3 * copy));
                next.emplace_back(u | shifted, e * eta);
            }
        }
        tensor = std::move(next);
    }
    std::sort(tensor.begin(), tensor.end());
    for (const auto& [u, e] : tensor) {
        if (!eqv(hypothesis_coefficient(0, u), {e, 0}) ||
            !eqv(hypothesis_coefficient(1, u), {-e, 0})) {
            throw std::logic_error("Factored tensor coefficient mismatch");
        }
        std::cout << u << ',' << e << ",0," << -e << ",0\n";
    }
}

struct Key {
    std::array<uint64_t, KEY_WORDS> words{};
};

inline bool operator==(const Key& x, const Key& y) { return x.words == y.words; }
inline bool is_zero_key(const Key& x) {
    return std::all_of(x.words.begin(), x.words.end(), [](uint64_t w) { return w == 0; });
}

uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

uint64_t hash_key(const Key& key) {
    uint64_t hash = 0x9e3779b97f4a7c15ULL;
    for (uint64_t word : key.words) hash = mix64(hash ^ mix64(word));
    return hash;
}

Key pack_rows(const std::vector<int>& rows) {
    if (rows.size() > N) throw std::logic_error("Too many stabilizer rows");
    Key key;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i] <= 0 || rows[i] >= (1 << SLOTBITS)) {
            throw std::logic_error("Invalid signed stabilizer row");
        }
        const size_t bit = i * SLOTBITS;
        const size_t word = bit / 64;
        const unsigned offset = bit % 64;
        const uint64_t value = static_cast<uint32_t>(rows[i]);
        key.words[word] |= value << offset;
        if (offset + SLOTBITS > 64) key.words[word + 1] |= value >> (64 - offset);
    }
    return key;
}

int sign_in_code(const std::vector<int>& code, int unsigned_label) {
    for (int p : code) {
        if ((p & (UMAX - 1)) == unsigned_label) return (p >> NB) & 1;
    }
    throw std::logic_error("Missing sign for row in stabilizer code");
}

Key canonical_key(const std::vector<int>& code) {
    const int rank = std::countr_zero(static_cast<unsigned>(code.size()));
    std::vector<int> rows;
    rows.reserve(code.size() - 1);
    for (int p : code) {
        const int u = p & (UMAX - 1);
        if (u != 0) rows.push_back(u);
    }

    int row = 0;
    for (int col = NB - 1; col >= 0 && row < rank; --col) {
        int pivot = -1;
        for (int i = row; i < static_cast<int>(rows.size()); ++i) {
            if ((rows[i] >> col) & 1) {
                pivot = i;
                break;
            }
        }
        if (pivot < 0) continue;
        std::swap(rows[row], rows[pivot]);
        for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
            if (i != row && ((rows[i] >> col) & 1)) rows[i] ^= rows[row];
        }
        ++row;
    }
    rows.resize(rank);

    std::vector<int> signed_rows;
    signed_rows.reserve(rank);
    for (int u : rows) signed_rows.push_back(u | (sign_in_code(code, u) << NB));
    return pack_rows(signed_rows);
}

std::vector<int> basis_from_key(const Key& key) {
    const uint64_t mask = (uint64_t{1} << SLOTBITS) - 1;
    std::vector<int> basis;
    for (int i = 0; i < N; ++i) {
        const size_t bit = static_cast<size_t>(i) * SLOTBITS;
        const size_t word = bit / 64;
        const unsigned offset = bit % 64;
        uint64_t value = key.words[word] >> offset;
        if (offset + SLOTBITS > 64) value |= key.words[word + 1] << (64 - offset);
        const int p = static_cast<int>(value & mask);
        if (p == 0) break;
        basis.push_back(p);
    }
    return basis;
}

std::vector<int> code_from_key(const Key& key) {
    const auto basis = basis_from_key(key);
    std::vector<int> code{0};
    for (int p : basis) {
        const int q = p & (UMAX - 1);
        const int sign = (p >> NB) & 1;
        code = extend_code(code, q, sign, N);
    }
    return code;
}

bool insert_linear(std::array<int, NB>& basis, int v) {
    for (int bit = NB - 1; bit >= 0; --bit) {
        if (((v >> bit) & 1) == 0) continue;
        if (basis[bit] != 0) {
            v ^= basis[bit];
        } else {
            basis[bit] = v;
            return true;
        }
    }
    return false;
}

std::vector<int> nullspace_basis(const std::vector<int>& constraints) {
    std::vector<int> rows = constraints;
    const int m = static_cast<int>(rows.size());
    int row = 0;
    std::vector<int> pivot_columns;

    for (int col = NB - 1; col >= 0 && row < m; --col) {
        int pivot = -1;
        for (int i = row; i < m; ++i) {
            if ((rows[i] >> col) & 1) {
                pivot = i;
                break;
            }
        }
        if (pivot < 0) continue;
        std::swap(rows[row], rows[pivot]);
        for (int i = 0; i < m; ++i) {
            if (i != row && ((rows[i] >> col) & 1)) rows[i] ^= rows[row];
        }
        pivot_columns.push_back(col);
        ++row;
    }
    rows.resize(row);

    std::array<bool, NB> is_pivot{};
    for (int c : pivot_columns) is_pivot[c] = true;

    std::vector<int> result;
    for (int free_column = 0; free_column < NB; ++free_column) {
        if (is_pivot[free_column]) continue;
        int v = 1 << free_column;
        for (int i = 0; i < row; ++i) {
            const int pivot = pivot_columns[i];
            if (parity_u(static_cast<unsigned>(rows[i] & v))) v |= 1 << pivot;
        }
        result.push_back(v);
    }
    return result;
}

std::vector<int> logical_complement(const Key& key) {
    const auto signed_basis = basis_from_key(key);
    std::vector<int> stabilizer_unsigned;
    std::vector<int> constraints;
    const int mask = (1 << N) - 1;

    for (int p : signed_basis) {
        const int u = p & (UMAX - 1);
        stabilizer_unsigned.push_back(u);
        const int x = u & mask;
        const int z = u >> N;
        constraints.push_back(z | (x << N));
    }

    const auto null_basis = nullspace_basis(constraints);
    std::array<int, NB> linear_basis{};
    for (int u : stabilizer_unsigned) insert_linear(linear_basis, u);

    std::vector<int> complement;
    for (int v : null_basis) {
        if (insert_linear(linear_basis, v)) complement.push_back(v);
    }

    const int expected = 2 * (N - static_cast<int>(stabilizer_unsigned.size()));
    if (static_cast<int>(complement.size()) != expected) {
        throw std::logic_error("Logical complement has incorrect dimension");
    }

    return complement;
}

uint64_t pack_value(Val value) {
    return static_cast<uint32_t>(value.a) |
           (static_cast<uint64_t>(static_cast<uint32_t>(value.b)) << 32);
}

Val unpack_value(uint64_t packed) {
    return {static_cast<int32_t>(packed & 0xffffffffULL),
            static_cast<int32_t>(packed >> 32)};
}

class FlatMemo {
  public:
    static size_t checked_capacity(int capacity_power) {
        if (capacity_power < 10 || capacity_power > 31) {
            throw std::invalid_argument("capacity power must be in [10,31]");
        }
        return size_t{1} << capacity_power;
    }

    explicit FlatMemo(int capacity_power)
        : capacity_(checked_capacity(capacity_power)),
          mask_(capacity_ - 1),
          keys_(capacity_),
          values_(capacity_),
          stamps_(capacity_, 0) {}

    void begin_epoch() {
        ++epoch_;
        count_ = 0;
        if (epoch_ == 0) {
            std::fill(stamps_.begin(), stamps_.end(), 0);
            epoch_ = 1;
        }
    }

    bool get(const Key& key, Val& value) const {
        size_t index = hash_key(key) & mask_;
        while (stamps_[index] == epoch_) {
            if (keys_[index] == key) {
                value = unpack_value(values_[index]);
                return true;
            }
            index = (index + 1) & mask_;
        }
        return false;
    }

    void put(const Key& key, Val value) {
        if ((count_ + 1) * 10 > capacity_ * 8) {
            throw std::runtime_error(
                "Flat memo exceeded 80% load. Increase --capacity-power.");
        }
        size_t index = hash_key(key) & mask_;
        while (stamps_[index] == epoch_) {
            if (keys_[index] == key) {
                values_[index] = pack_value(value);
                return;
            }
            index = (index + 1) & mask_;
        }
        stamps_[index] = epoch_;
        keys_[index] = key;
        values_[index] = pack_value(value);
        ++count_;
    }

    size_t size() const { return count_; }
    size_t capacity() const { return capacity_; }

    double estimated_gib() const {
        const long double bytes =
            static_cast<long double>(capacity_) *
            (sizeof(Key) + sizeof(uint64_t) + sizeof(uint32_t));
        return static_cast<double>(bytes / (1024.0L * 1024.0L * 1024.0L));
    }

  private:
    size_t capacity_;
    size_t mask_;
    std::vector<Key> keys_;
    std::vector<uint64_t> values_;
    std::vector<uint32_t> stamps_;
    uint32_t epoch_ = 1;
    size_t count_ = 0;
};

struct BranchStats {
    uint64_t completed = 0;
    std::array<uint64_t, N + 1> by_rank{};
    double seconds = 0.0;
};

class BranchSolver {
  public:
    BranchSolver(int capacity_power, uint64_t progress_every, uint64_t node_limit,
                 int worker_id, const std::atomic<bool>* cancelled = nullptr)
        : memo_(capacity_power),
          progress_every_(progress_every),
          node_limit_(node_limit),
          worker_id_(worker_id),
          cancelled_(cancelled) {}

    Val solve_branch(int q, int outcome_sign, BranchStats& stats) {
        memo_.begin_epoch();
        completed_ = 0;
        by_rank_.fill(0);
        branch_started_ = std::chrono::steady_clock::now();

        std::vector<int> root_code{0};
        const auto child = extend_code(root_code, q, outcome_sign, N);
        const Val answer = solve(canonical_key(child));

        stats.completed = completed_;
        stats.by_rank = by_rank_;
        stats.seconds = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - branch_started_)
                            .count();
        return answer;
    }

    double memo_gib() const { return memo_.estimated_gib(); }

    // Independent regression tests can solve a small, high-rank code without
    // launching an exhaustive all-root calculation.
    Val solve_code_for_test(const std::vector<int>& code) {
        memo_.begin_epoch();
        completed_ = 0;
        by_rank_.fill(0);
        branch_started_ = std::chrono::steady_clock::now();
        return solve(canonical_key(code));
    }

  private:
    Val immediate_stop(const std::vector<int>& code, int rank, Val* upper_out) const {
        Val sums[2] = {{0, 0}, {0, 0}};
        for (int p : code) {
            const int u = p & (UMAX - 1);
            const int sign = (p >> NB) & 1;
            for (int s = 0; s < 2; ++s) {
                Val coefficient = hypothesis_coefficient(s, u);
                if (sign) coefficient = negv(coefficient);
                sums[s] = addv(sums[s], coefficient);
            }
        }
        const int factor = 1 << (N - rank);
        const Val v0 = {sums[0].a * factor, sums[0].b * factor};
        const Val v1 = {sums[1].a * factor, sums[1].b * factor};
        if (upper_out) *upper_out = addv(v0, v1);
        return lessv(v0, v1) ? v1 : v0;
    }

    Val solve(const Key& key) {
        if (cancelled_ && cancelled_->load(std::memory_order_relaxed)) {
            throw std::runtime_error("Search cancelled after another worker failed");
        }
        Val cached;
        if (memo_.get(key, cached)) return cached;

        if (node_limit_ != 0 && completed_ >= node_limit_) {
            throw std::runtime_error("Node limit reached (smoke-test termination)");
        }

        const auto code = code_from_key(key);
        const int rank = std::countr_zero(static_cast<unsigned>(code.size()));
        Val upper;
        Val best = immediate_stop(code, rank, &upper);

        if (rank < N && lessv(best, upper)) {
            // Enumerate nonzero quotient vectors lazily in Gray-code order:
            // just one XOR per measurement and O(N) scratch per recursion level.
            const auto complement = logical_complement(key);
            const int count = 1 << static_cast<int>(complement.size());
            int q = 0;
            for (int index = 1; index < count; ++index) {
                q ^= complement[std::countr_zero(static_cast<unsigned>(index))];
                const auto plus = extend_code(code, q, 0, N);
                const auto minus = extend_code(code, q, 1, N);
                const Val candidate = addv(solve(canonical_key(plus)),
                                           solve(canonical_key(minus)));
                if (lessv(best, candidate)) best = candidate;
                if (eqv(best, upper)) break;
            }
        }

        if (node_limit_ != 0 && completed_ >= node_limit_) {
            throw std::runtime_error("Node limit reached (smoke-test termination)");
        }
        memo_.put(key, best);
        ++completed_;
        ++by_rank_[rank];

        if (progress_every_ != 0 && completed_ % progress_every_ == 0) {
            const double seconds = std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() - branch_started_)
                                       .count();
            std::lock_guard<std::mutex> lock(io_mutex);
            std::cerr << "worker=" << worker_id_ << " branch_completed=" << completed_
                      << " rank=" << rank << " memo=" << memo_.size()
                      << " nodeV=" << val_string(best) << " dec=" << decv(best)
                      << " elapsed=" << seconds << "s\n";
        }
        return best;
    }

    FlatMemo memo_;
    uint64_t progress_every_;
    uint64_t node_limit_;
    int worker_id_;
    const std::atomic<bool>* cancelled_;
    uint64_t completed_ = 0;
    std::array<uint64_t, N + 1> by_rank_{};
    std::chrono::steady_clock::time_point branch_started_;

  public:
    static std::mutex io_mutex;
};

std::mutex BranchSolver::io_mutex;

struct RootResult {
    int q = 0;
    Val value{};
    Val plus{};
    Val minus{};
    uint64_t nodes_plus = 0;
    uint64_t nodes_minus = 0;
    double seconds = 0.0;
};

struct Options {
    int threads = 1;
    int q_start = 1;
    int q_end = UMAX - 1;
    int capacity_power = DEFAULT_CAPACITY_POWER;
    uint64_t progress_every = 500'000;
    uint64_t node_limit = 0;
    std::string q_file;
    std::string output = "e8_xor_k" + std::to_string(COPIES) + "_root_results.csv";
    bool resume = true;
    bool self_test = false;
    bool dump_hypotheses = false;
    std::vector<std::string> combine_files;
};

std::vector<int> read_q_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Could not open q-file: " + path);
    std::vector<int> q_values;
    std::string line;
    while (std::getline(input, line)) {
        const auto hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        std::istringstream parser(line);
        int q;
        while (parser >> q) {
            if (q <= 0 || q >= UMAX) {
                throw std::runtime_error("q-file contains an invalid Pauli label");
            }
            q_values.push_back(q);
        }
    }
    std::sort(q_values.begin(), q_values.end());
    q_values.erase(std::unique(q_values.begin(), q_values.end()), q_values.end());
    return q_values;
}

std::unordered_map<int, RootResult> read_results_file(const std::string& path) {
    std::unordered_map<int, RootResult> results;
    std::ifstream input(path);
    if (!input) return results;
    if (input.peek() == std::ifstream::traits_type::eof()) return results;
    bool denominator_ok = false, copies_ok = false, labels_ok = false;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            auto metadata = [&](const std::string& prefix, const std::string& expected,
                                bool& present) {
                if (line.rfind(prefix, 0) != 0) return;
                if (line != expected) {
                    throw std::runtime_error("Conflicting CSV metadata: " + path);
                }
                present = true;
            };
            metadata("# VALUE_DEN=", "# VALUE_DEN=" + std::to_string(VALUE_DEN), denominator_ok);
            metadata("# COPIES=", "# COPIES=" + std::to_string(COPIES), copies_ok);
            metadata("# LABEL_ORDER=", "# LABEL_ORDER=python_v1", labels_ok);
            continue;
        }
        if (line.rfind("q,", 0) == 0) continue;
        if (!denominator_ok || !copies_ok || !labels_ok) {
            throw std::runtime_error("Incompatible CSV metadata (copies, denominator or label order): " + path);
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream parser(line);
        RootResult result;
        std::string trailing;
        if (!(parser >> result.q >> result.value.a >> result.value.b >> result.plus.a >>
              result.plus.b >> result.minus.a >> result.minus.b >> result.nodes_plus >>
              result.nodes_minus >> result.seconds) || (parser >> trailing)) {
            throw std::runtime_error("Malformed result row: " + path);
        }
        // This family is rational. Validate the completed-root records before
        // allowing either resume or --combine to certify a full calculation.
        if (result.q <= 0 || result.q >= UMAX || result.value.b != 0 ||
            result.plus.b != 0 || result.minus.b != 0 ||
            result.plus.a < 0 || result.plus.a > VALUE_DEN / 2 ||
            result.minus.a < 0 || result.minus.a > VALUE_DEN / 2 ||
            result.value.a < VALUE_DEN / 2 || result.value.a > VALUE_DEN ||
            !eqv(result.value, addv(result.plus, result.minus)) ||
            !std::isfinite(result.seconds) || result.seconds < 0) {
            throw std::runtime_error("Invalid completed-root result: " + path);
        }
        const auto [it, inserted] = results.emplace(result.q, result);
        if (!inserted && (!eqv(it->second.value, result.value) ||
                          !eqv(it->second.plus, result.plus) ||
                          !eqv(it->second.minus, result.minus))) {
            throw std::runtime_error("Conflicting duplicate root result: " + path);
        }
    }
    if (!denominator_ok || !copies_ok || !labels_ok) {
        throw std::runtime_error("Incompatible CSV metadata (copies, denominator or label order): " + path);
    }
    return results;
}

void append_result(std::ofstream& output, const RootResult& result) {
    output << result.q << ',' << result.value.a << ',' << result.value.b << ','
           << result.plus.a << ',' << result.plus.b << ',' << result.minus.a << ','
           << result.minus.b << ',' << result.nodes_plus << ',' << result.nodes_minus
           << ',' << std::setprecision(17) << result.seconds << '\n';
    output.flush();
    if (!output) throw std::runtime_error("Could not write completed-root result");
}

void print_summary(const std::unordered_map<int, RootResult>& results,
                   size_t required_count, bool /*representatives_assumed*/) {
    Val best = {VALUE_DEN / 2, 0};
    int best_q = 0;
    for (const auto& [q, result] : results) {
        if (lessv(best, result.value)) { best = result.value; best_q = q; }
    }
    std::cout << "\n=== root summary ===\n"
              << "completed first measurements = " << results.size() << " / "
              << required_count << "\n"
              << "best q = " << best_q << "\n"
              << "best value = " << val_string(best) << " = " << decv(best) << "\n";
    // A completed subset is only a lower bound. No user-provided orbit claim
    // or selected range may produce the verified-completion sentinel.
    bool full = results.size() == static_cast<size_t>(UMAX - 1);
    for (int q = 1; full && q < UMAX; ++q) full = results.contains(q);
    if (full) {
        std::cout << "RESULT_NUM " << best.a << ' ' << best.b << " DEN "
                  << VALUE_DEN << "\nExact raw root search complete.\n";
    } else {
        std::cout << "Partial run: this is only a certified lower bound.\n";
    }
}

void combine_files(const Options& options) {
    std::unordered_map<int, RootResult> all;
    for (const auto& path : options.combine_files) {
        if (!std::ifstream(path)) throw std::runtime_error("Could not open combined CSV: " + path);
        auto part = read_results_file(path);
        for (const auto& [q, result] : part) {
            const auto [it, inserted] = all.emplace(q, result);
            if (!inserted && (!eqv(it->second.value, result.value) ||
                              !eqv(it->second.plus, result.plus) ||
                              !eqv(it->second.minus, result.minus))) {
                throw std::runtime_error("Conflicting duplicate root across CSV files");
            }
        }
    }
    print_summary(all, UMAX - 1, false);
}

void self_test() {
    build_hypotheses();
    if (!eqv(hypothesis_coefficient(0, 0), {COEFF_DEN, 0}) ||
        !eqv(hypothesis_coefficient(1, 0), {COEFF_DEN, 0})) {
        throw std::logic_error("Identity coefficient test failed");
    }
    std::vector<int> root{0};
    if (!is_zero_key(canonical_key(root))) throw std::logic_error("Root key is not zero");
    if (canonical_key(extend_code(root, 1, 0, N)) ==
        canonical_key(extend_code(root, 1, 1, N))) {
        throw std::logic_error("Opposite measurement outcomes have identical keys");
    }
    // Fill every row, including row fields crossing 64/128/192/256-bit word
    // boundaries. Every one-bit change must survive packing and unpacking.
    std::vector<int> rows;
    for (int i = 0; i < N; ++i) rows.push_back((1 << i) | (1 << NB));
    const Key packed = pack_rows(rows);
    if (basis_from_key(packed) != rows) throw std::logic_error("Wide key roundtrip failed");
    for (int row = 0; row < N; ++row) {
        for (int bit = 0; bit < SLOTBITS; ++bit) {
            auto changed = rows;
            changed[row] ^= 1 << bit;
            const Key other = pack_rows(changed);
            if (other == packed || basis_from_key(other) != changed) {
                throw std::logic_error("Wide key bit collision");
            }
        }
    }
    const auto code = code_from_key(packed);
    if (code_from_key(canonical_key(code)) != code) {
        throw std::logic_error("Stabilizer-code canonical roundtrip failed");
    }
    BranchSolver smoke(12, 0, 1000, 0);
    BranchStats stats;
    try { (void)smoke.solve_branch(1, 0, stats); }
    catch (const std::runtime_error& error) {
        if (std::string(error.what()).find("Node limit reached") == std::string::npos) throw;
    }
    std::cout << "self-test passed; copies=" << COPIES << " qubits=" << N
              << " key_words=" << KEY_WORDS << " denominator=" << VALUE_DEN
              << "; bounded recursion checked\n";
}

int integer_option(const std::string& text) {
    size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != text.size()) throw std::invalid_argument("Invalid integer option: " + text);
    return value;
}

uint64_t count_option(const std::string& text) {
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument("Count options must be nonnegative");
    }
    size_t consumed = 0;
    const uint64_t value = std::stoull(text, &consumed);
    if (consumed != text.size()) throw std::invalid_argument("Invalid count option: " + text);
    return value;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(std::string(name) + " needs a value");
            return argv[++i];
        };

        if (arg == "--threads") {
            options.threads = integer_option(require_value("--threads"));
        } else if (arg == "--q-start") {
            options.q_start = integer_option(require_value("--q-start"));
        } else if (arg == "--q-end") {
            options.q_end = integer_option(require_value("--q-end"));
        } else if (arg == "--q-file") {
            options.q_file = require_value("--q-file");
        } else if (arg == "--output") {
            options.output = require_value("--output");
        } else if (arg == "--capacity-power") {
            options.capacity_power = integer_option(require_value("--capacity-power"));
        } else if (arg == "--progress-every") {
            options.progress_every = count_option(require_value("--progress-every"));
        } else if (arg == "--node-limit") {
            options.node_limit = count_option(require_value("--node-limit"));
        } else if (arg == "--no-resume") {
            options.resume = false;
        } else if (arg == "--self-test") {
            options.self_test = true;
        } else if (arg == "--dump-hypotheses") {
            options.dump_hypotheses = true;
        } else if (arg == "--combine") {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                options.combine_files.push_back(argv[++i]);
            }
            if (options.combine_files.empty()) {
                throw std::invalid_argument("--combine needs one or more CSV files");
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: e8_xor_k" << COPIES << " [options]\n\n"
                << "  --threads N             Parallel root jobs (default 1)\n"
                << "  --q-start A             First unsigned Pauli label (default 1)\n"
                << "  --q-end B               Last unsigned Pauli label (default " << UMAX - 1 << ")\n"
                << "  --q-file FILE           Use listed roots (a subset gives a lower bound)\n"
                << "  --output FILE           Append/resume CSV results\n"
                << "  --capacity-power P      Per-worker table 2^P slots, P=10..31 (default "
                << DEFAULT_CAPACITY_POWER << ")\n"
                << "  --progress-every N      Branch-node progress cadence\n"
                << "  --node-limit N          Smoke test: stop each branch after N solved nodes\n"
                << "  --no-resume             Ignore existing output CSV\n"
                << "  --self-test             Run algebra/key/smoke checks\n"
                << "  --dump-hypotheses       Print exact tensor-support coefficients\n"
                << "  --combine FILE...       Combine raw range-result CSV files\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown option: " + arg);
        }
    }

    if (options.threads <= 0) throw std::invalid_argument("threads must be positive");
    (void)FlatMemo::checked_capacity(options.capacity_power);
    if (options.q_start < 1 || options.q_end >= UMAX || options.q_start > options.q_end) {
        throw std::invalid_argument("Invalid q range");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (!options.combine_files.empty()) {
            combine_files(options);
            return 0;
        }
        if (options.self_test) {
            self_test();
            return 0;
        }
        if (options.dump_hypotheses) {
            dump_hypotheses();
            return 0;
        }

        build_hypotheses();

        std::vector<int> q_values;
        if (!options.q_file.empty()) {
            q_values = read_q_file(options.q_file);
        } else {
            for (int q = options.q_start; q <= options.q_end; ++q) q_values.push_back(q);
        }

        if (q_values.empty()) throw std::invalid_argument("No root measurements selected");

        auto existing = options.resume ? read_results_file(options.output)
                                       : std::unordered_map<int, RootResult>{};
        std::vector<int> pending;
        for (int q : q_values) {
            if (existing.find(q) == existing.end()) pending.push_back(q);
        }

        const size_t memo_capacity = size_t{1} << options.capacity_power;
        const long double memo_bytes = static_cast<long double>(memo_capacity) *
            (sizeof(Key) + sizeof(uint64_t) + sizeof(uint32_t));
        const double gib_per_worker = static_cast<double>(
            memo_bytes / (1024.0L * 1024.0L * 1024.0L));
        std::cout << "problem = " << COPIES << "-copy E8 XOR/sign parity\n"
                  << "qubits = " << N << "\n"
                  << "value denominator = " << VALUE_DEN << "\n"
                  << "requested root measurements = " << q_values.size() << "\n"
                  << "already completed = " << (q_values.size() - pending.size()) << "\n"
                  << "pending = " << pending.size() << "\n"
                  << "threads = " << options.threads << "\n"
                  << "flat memo capacity = 2^" << options.capacity_power << " slots\n"
                  << "estimated memo memory per worker = " << std::fixed
                  << std::setprecision(2) << gib_per_worker << " GiB\n"
                  << "estimated total memo memory = " << gib_per_worker * options.threads
                  << " GiB\n" << std::defaultfloat << std::setprecision(17) << std::flush;

        if (pending.empty()) {
            std::unordered_map<int, RootResult> selected_results;
            for (int q : q_values) {
                const auto it = existing.find(q);
                if (it != existing.end()) selected_results.emplace(q, it->second);
            }
            print_summary(selected_results, q_values.size(), !options.q_file.empty());
            return 0;
        }

        std::ofstream output;
        {
            std::ifstream previous_output(options.output);
            const bool file_exists = previous_output &&
                previous_output.peek() != std::ifstream::traits_type::eof();
            previous_output.close();
            const auto mode = options.resume ? std::ios::app : std::ios::trunc;
            output.open(options.output, mode);
            if (!output) throw std::runtime_error("Could not open output file");
            if (!file_exists || !options.resume) {
                output << "# stabdisc E8 XOR, raw root split\n"
                       << "# COPIES=" << COPIES << "\n"
                       << "# LABEL_ORDER=python_v1\n"
                       << "# VALUE_DEN=" << VALUE_DEN << "\n"
                       << "q,a,b,plus_a,plus_b,minus_a,minus_b,nodes_plus,nodes_minus,seconds\n";
                output.flush();
                if (!output) throw std::runtime_error("Could not write result CSV header");
            }
        }

        std::mutex result_mutex;
        std::atomic<size_t> next_index{0};
        std::atomic<size_t> finished{0};
        std::atomic<bool> cancelled{false};
        std::exception_ptr worker_error;
        const auto all_started = std::chrono::steady_clock::now();

        auto worker = [&](int worker_id) {
          try {
            BranchSolver solver(options.capacity_power, options.progress_every,
                                options.node_limit, worker_id, &cancelled);
            while (!cancelled.load(std::memory_order_relaxed)) {
                const size_t index = next_index.fetch_add(1);
                if (index >= pending.size()) break;
                const int q = pending[index];
                const auto started = std::chrono::steady_clock::now();

                BranchStats plus_stats, minus_stats;
                RootResult result;
                result.q = q;
                result.plus = solver.solve_branch(q, 0, plus_stats);
                result.minus = solver.solve_branch(q, 1, minus_stats);
                result.value = addv(result.plus, result.minus);
                result.nodes_plus = plus_stats.completed;
                result.nodes_minus = minus_stats.completed;
                result.seconds = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - started)
                                     .count();

                {
                    std::lock_guard<std::mutex> lock(result_mutex);
                    existing[q] = result;
                    append_result(output, result);
                    const size_t done = ++finished;
                    const double total_seconds = std::chrono::duration<double>(
                                                   std::chrono::steady_clock::now() - all_started)
                                                   .count();
                    std::cout << "root_done=" << done << '/' << pending.size()
                              << " q=" << q << " candidate=" << val_string(result.value)
                              << " dec=" << decv(result.value)
                              << " nodes=(" << result.nodes_plus << ','
                              << result.nodes_minus << ") seconds=" << result.seconds
                              << " total_elapsed=" << total_seconds << "s\n" << std::flush;
                }
            }
          } catch (...) {
            std::lock_guard<std::mutex> lock(result_mutex);
            if (!worker_error) worker_error = std::current_exception();
            cancelled.store(true, std::memory_order_relaxed);
          }
        };

        std::vector<std::thread> threads;
        threads.reserve(options.threads);
        try {
            const size_t worker_count = std::min(static_cast<size_t>(options.threads), pending.size());
            for (size_t t = 0; t < worker_count; ++t) threads.emplace_back(worker, static_cast<int>(t));
        } catch (...) {
            cancelled.store(true, std::memory_order_relaxed);
            for (auto& thread : threads) thread.join();
            throw;
        }
        for (auto& thread : threads) thread.join();
        if (worker_error) std::rethrow_exception(worker_error);

        std::unordered_map<int, RootResult> selected_results;
        for (int q : q_values) {
            const auto it = existing.find(q);
            if (it != existing.end()) selected_results.emplace(q, it->second);
        }
        print_summary(selected_results, q_values.size(), !options.q_file.empty());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
