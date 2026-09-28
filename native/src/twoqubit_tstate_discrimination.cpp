#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

// Exact adaptive-stabilizer discrimination of
//
// rho_0 = ( |0+><0+| + |+0><+0| ) / 2,
// rho_1 = ( |11><11| + |--><--| ) / 2,
//
// assisted by TSTATES copies of
// |T> = (|0> + exp(i pi/4)|1>)/sqrt(2).
//
// Compile separately with -DTSTATES=0,1,2,3,4.  The total number of qubits is
// N=2+TSTATES.  The program performs an exact raw search over all possible
// first Pauli measurements and then solves each fixed-outcome branch by Bellman
// recursion with memoization.  First measurements are independent parallel
// jobs and completed roots are checkpointed to CSV.
//
// The computed resource is T-state assistance.  It lower-bounds the strict
// unitary T-count needed for perfect discrimination and coincides with it only
// after the injection/adaptivity model has been specified.

#ifndef TSTATES
#define TSTATES 0
#endif

static_assert(TSTATES >= 0 && TSTATES <= 4,
              "This raw implementation supports TSTATES=0,1,2,3,4.");

namespace {

constexpr int M = TSTATES;
constexpr int N = 2 + M;
constexpr int NB = 2 * N;
constexpr int UMAX = 1 << NB;
constexpr int SLOTBITS = NB + 1;
constexpr int NUM_HYP = 2;

// System expectations have denominator 2.  Multiplication by up to M factors
// 1/sqrt(2) fits in the common Q(sqrt(2)) denominator below.
constexpr int EXPECT_DEN = 1 << (1 + (M + 1) / 2);
constexpr int VALUE_DEN = NUM_HYP * EXPECT_DEN * (1 << N);

constexpr uint64_t branch_projector_bound() {
    // Total numbers of signed stabilizer-code projectors on 1,...,5 qubits.
    if constexpr (N - 1 == 1) return 7ULL;
    if constexpr (N - 1 == 2) return 91ULL;
    if constexpr (N - 1 == 3) return 2'467ULL;
    if constexpr (N - 1 == 4) return 150'451ULL;
    if constexpr (N - 1 == 5) return 21'555'667ULL;
    return 0ULL;
}
constexpr uint64_t EXPECTED_BRANCH_PROJECTORS = branch_projector_bound();

constexpr int default_capacity_power() {
    if constexpr (N == 2) return 4;   // 16 slots
    if constexpr (N == 3) return 8;   // 256 slots
    if constexpr (N == 4) return 12;  // 4096 slots
    if constexpr (N == 5) return 18;  // 262144 slots
    if constexpr (N == 6) return 25;  // 33,554,432 slots; ~0.875 GiB/worker
    return 20;
}
constexpr int DEFAULT_CAPACITY_POWER = default_capacity_power();

struct Val {
    int32_t a = 0;
    int32_t b = 0;
};  // (a+b sqrt(2))/VALUE_DEN

inline Val addv(Val x, Val y) { return {x.a + y.a, x.b + y.b}; }
inline Val negv(Val x) { return {-x.a, -x.b}; }
inline bool eqv(Val x, Val y) { return x.a == y.a && x.b == y.b; }

int sign_exact(int64_t a, int64_t b) {
    if (b == 0) return (a > 0) - (a < 0);
    if (a == 0) return (b > 0) - (b < 0);
    if (a > 0 && b > 0) return 1;
    if (a < 0 && b < 0) return -1;
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
    return (static_cast<double>(x.a) +
            static_cast<double>(x.b) * std::sqrt(2.0)) /
           VALUE_DEN;
}

std::string val_string(Val x) {
    std::ostringstream out;
    out << "(" << x.a << " + " << x.b << " sqrt(2))/" << VALUE_DEN;
    return out.str();
}

// Local labels I=0, X=1, Z=2, Y=3.  PH is e in AB=i^e C.
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

std::array<std::array<Val, UMAX>, 2> HYP{};

int system_expectation_numerator(int hypothesis, int local_index) {
    // Returns 2 Tr(P rho_s), so all values are integers.
    // local_index = p0 + 4*p1 with I=0,X=1,Z=2,Y=3.
    static const std::array<int, 16> rho0 = [] {
        std::array<int, 16> v{};
        v[0] = 2;                  // II
        v[1] = v[4] = 1;          // XI, IX
        v[2] = v[8] = 1;          // ZI, IZ
        v[2 + 4 * 1] = 1;         // ZX
        v[1 + 4 * 2] = 1;         // XZ
        return v;
    }();
    static const std::array<int, 16> rho1 = [] {
        std::array<int, 16> v{};
        v[0] = 2;                  // II
        v[1] = v[4] = -1;         // XI, IX
        v[2] = v[8] = -1;         // ZI, IZ
        v[1 + 4 * 1] = 1;         // XX
        v[2 + 4 * 2] = 1;         // ZZ
        return v;
    }();
    return hypothesis == 0 ? rho0[local_index] : rho1[local_index];
}

void build_hypotheses() {
    for (auto& row : HYP) {
        for (auto& value : row) value = {0, 0};
    }

    const int mask = (1 << N) - 1;
    for (int u = 0; u < UMAX; ++u) {
        const int x = u & mask;
        const int z = u >> N;
        const int p0 = ((x >> 0) & 1) | (((z >> 0) & 1) << 1);
        const int p1 = ((x >> 1) & 1) | (((z >> 1) & 1) << 1);
        const int local_index = p0 + 4 * p1;

        int magic_weight = 0;
        bool zero = false;
        for (int j = 0; j < M; ++j) {
            const int q = 2 + j;
            const int local = ((x >> q) & 1) | (((z >> q) & 1) << 1);
            if (local == 2) {  // Z expectation of |T> is zero.
                zero = true;
                break;
            }
            if (local == 1 || local == 3) ++magic_weight;  // X or Y.
        }
        if (zero) continue;

        for (int s = 0; s < 2; ++s) {
            const int sys_num = system_expectation_numerator(s, local_index);
            if (sys_num == 0) continue;

            if ((magic_weight & 1) == 0) {
                const int exponent = 1 + magic_weight / 2;
                const int denominator = 1 << exponent;
                const int scale = EXPECT_DEN / denominator;
                HYP[s][u] = {sys_num * scale, 0};
            } else {
                const int exponent = 1 + (magic_weight + 1) / 2;
                const int denominator = 1 << exponent;
                const int scale = EXPECT_DEN / denominator;
                HYP[s][u] = {0, sys_num * scale};
            }
        }
    }

    if (!eqv(HYP[0][0], {EXPECT_DEN, 0}) ||
        !eqv(HYP[1][0], {EXPECT_DEN, 0})) {
        throw std::logic_error("Hypothesis identity expectation is not one");
    }
}

struct Key {
    uint64_t lo = 0;
    uint64_t hi = 0;
};

inline bool operator==(const Key& x, const Key& y) {
    return x.lo == y.lo && x.hi == y.hi;
}
inline bool is_zero_key(const Key& x) { return x.lo == 0 && x.hi == 0; }

uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

uint64_t hash_key(const Key& key) {
    return mix64(key.lo ^ std::rotl(key.hi * 0x9e3779b97f4a7c15ULL, 27));
}

Key pack_rows(const std::vector<int>& rows) {
    unsigned __int128 packed = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        packed |= static_cast<unsigned __int128>(static_cast<uint32_t>(rows[i]))
                  << (i * SLOTBITS);
    }
    return {static_cast<uint64_t>(packed), static_cast<uint64_t>(packed >> 64)};
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
    const unsigned __int128 packed =
        static_cast<unsigned __int128>(key.lo) |
        (static_cast<unsigned __int128>(key.hi) << 64);
    const unsigned __int128 mask =
        (static_cast<unsigned __int128>(1) << SLOTBITS) - 1;
    std::vector<int> basis;
    for (int i = 0; i < N; ++i) {
        const int p = static_cast<int>((packed >> (i * SLOTBITS)) & mask);
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

std::vector<int> logical_representatives(const Key& key) {
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

    std::vector<int> representatives;
    representatives.reserve((1 << complement.size()) - 1);
    for (int bits = 1; bits < (1 << static_cast<int>(complement.size())); ++bits) {
        int q = 0;
        for (int i = 0; i < static_cast<int>(complement.size()); ++i) {
            if ((bits >> i) & 1) q ^= complement[i];
        }
        representatives.push_back(q);
    }
    return representatives;
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
    explicit FlatMemo(int capacity_power)
        : capacity_(size_t{1} << capacity_power),
          mask_(capacity_ - 1),
          keys_(capacity_),
          values_(capacity_),
          stamps_(capacity_, 0) {
        if (capacity_power < 4 || capacity_power > 31) {
            throw std::invalid_argument("capacity power must be in [4,31]");
        }
    }

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
        const long double bytes = static_cast<long double>(capacity_) *
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
                 int worker_id)
        : memo_(capacity_power),
          progress_every_(progress_every),
          node_limit_(node_limit),
          worker_id_(worker_id) {}

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

  private:
    Val immediate_stop(const std::vector<int>& code, int rank, Val* upper_out) const {
        Val sums[2] = {{0, 0}, {0, 0}};
        for (int p : code) {
            const int u = p & (UMAX - 1);
            const int sign = (p >> NB) & 1;
            for (int s = 0; s < 2; ++s) {
                Val coefficient = HYP[s][u];
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
            const auto representatives = logical_representatives(key);
            for (int q : representatives) {
                const auto plus = extend_code(code, q, 0, N);
                const auto minus = extend_code(code, q, 1, N);
                const Val candidate = addv(solve(canonical_key(plus)),
                                           solve(canonical_key(minus)));
                if (lessv(best, candidate)) best = candidate;
                if (eqv(best, upper)) break;
            }
        }

        memo_.put(key, best);
        ++completed_;
        ++by_rank_[rank];

        if (progress_every_ != 0 && completed_ % progress_every_ == 0) {
            const double seconds = std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() - branch_started_)
                                       .count();
            std::lock_guard<std::mutex> lock(io_mutex);
            std::cerr << "worker=" << worker_id_
                      << " branch_completed=" << completed_
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
    uint64_t progress_every = 100'000;
    uint64_t node_limit = 0;
    std::string q_file;
    std::string output = "twoqubit_tstate_m" + std::to_string(M) + "_roots.csv";
    bool resume = true;
    bool self_test = false;
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
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line.rfind("q,", 0) == 0) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream parser(line);
        RootResult result;
        if (parser >> result.q >> result.value.a >> result.value.b >> result.plus.a >>
                result.plus.b >> result.minus.a >> result.minus.b >> result.nodes_plus >>
                result.nodes_minus >> result.seconds) {
            results[result.q] = result;
        }
    }
    return results;
}

void append_result(std::ofstream& output, const RootResult& result) {
    output << result.q << ',' << result.value.a << ',' << result.value.b << ','
           << result.plus.a << ',' << result.plus.b << ',' << result.minus.a << ','
           << result.minus.b << ',' << result.nodes_plus << ',' << result.nodes_minus
           << ',' << std::setprecision(17) << result.seconds << '\n';
    output.flush();
}

void print_summary(const std::unordered_map<int, RootResult>& results,
                   size_t required_count, bool representatives_assumed) {
    Val best = {VALUE_DEN / 2, 0};  // Ignore input and guess.
    int best_q = 0;
    for (const auto& [q, result] : results) {
        if (lessv(best, result.value)) {
            best = result.value;
            best_q = q;
        }
    }

    std::cout << "\n=== root summary ===\n";
    std::cout << "T states = " << M << "\n";
    std::cout << "completed first measurements = " << results.size() << " / "
              << required_count << "\n";
    std::cout << "best q = " << best_q << "\n";
    std::cout << "best value = " << val_string(best) << " = "
              << std::setprecision(12) << decv(best) << "\n";
    if (results.size() == required_count) {
        std::cout << "RESULT_NUM " << best.a << ' ' << best.b << " DEN "
                  << VALUE_DEN << "\n";
        if (representatives_assumed) {
            std::cout << "Result is exact provided the q-file contains one representative "
                         "from every root-measurement orbit.\n";
        } else {
            std::cout << "Exact raw root search complete.\n";
        }
    } else {
        std::cout << "Partial run: this is only a certified lower bound.\n";
    }
}

void combine_files(const Options& options) {
    std::unordered_map<int, RootResult> all;
    for (const auto& path : options.combine_files) {
        auto part = read_results_file(path);
        all.insert(part.begin(), part.end());
    }
    print_summary(all, UMAX - 1, false);
}

void self_test() {
    build_hypotheses();
    if (!eqv(HYP[0][0], {EXPECT_DEN, 0}) ||
        !eqv(HYP[1][0], {EXPECT_DEN, 0})) {
        throw std::logic_error("Identity coefficient test failed");
    }

    std::vector<int> root{0};
    if (!is_zero_key(canonical_key(root))) {
        throw std::logic_error("Root key is not zero");
    }
    if (canonical_key(extend_code(root, 1, 0, N)) ==
        canonical_key(extend_code(root, 1, 1, N))) {
        throw std::logic_error("Opposite outcomes have identical keys");
    }

    if constexpr (M == 0) {
        BranchSolver solver(DEFAULT_CAPACITY_POWER, 0, 0, 0);
        Val best = {VALUE_DEN / 2, 0};
        for (int q = 1; q < UMAX; ++q) {
            BranchStats ps, ms;
            const Val candidate = addv(solver.solve_branch(q, 0, ps),
                                       solver.solve_branch(q, 1, ms));
            if (lessv(best, candidate)) best = candidate;
        }
        const Val expected = {7 * VALUE_DEN / 8, 0};
        if (!eqv(best, expected)) {
            throw std::logic_error("m=0 validation failed: expected 7/8");
        }
        std::cout << "self-test passed; p_success(m=0)=7/8\n";
    } else if constexpr (M <= 3) {
        BranchSolver smoke(DEFAULT_CAPACITY_POWER, 0, 0, 0);
        BranchStats ps, ms;
        const Val candidate = addv(smoke.solve_branch(1, 0, ps),
                                   smoke.solve_branch(1, 1, ms));
        if (lessv(candidate, {VALUE_DEN / 2, 0}) ||
            lessv({VALUE_DEN, 0}, candidate)) {
            throw std::logic_error("Smoke result lies outside [1/2,1]");
        }
        std::cout << "self-test passed; one complete root candidate = "
                  << val_string(candidate) << "\n";
    } else {
        // A complete six-qubit root can require tens of millions of Bellman
        // states.  The self-test deliberately stops after 5,000 solved nodes.
        BranchSolver smoke(20, 1000, 5000, 0);
        BranchStats stats;
        try {
            (void)smoke.solve_branch(1, 0, stats);
        } catch (const std::runtime_error& error) {
            const std::string message = error.what();
            if (message.find("Node limit reached") == std::string::npos) throw;
        }
        std::cout << "self-test passed; smoke recursion reached the requested node limit\n";
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string(name) + " needs a value");
            }
            return argv[++i];
        };

        if (arg == "--threads") {
            options.threads = std::stoi(require_value("--threads"));
        } else if (arg == "--q-start") {
            options.q_start = std::stoi(require_value("--q-start"));
        } else if (arg == "--q-end") {
            options.q_end = std::stoi(require_value("--q-end"));
        } else if (arg == "--q-file") {
            options.q_file = require_value("--q-file");
        } else if (arg == "--output") {
            options.output = require_value("--output");
        } else if (arg == "--capacity-power") {
            options.capacity_power = std::stoi(require_value("--capacity-power"));
        } else if (arg == "--progress-every") {
            options.progress_every = std::stoull(require_value("--progress-every"));
        } else if (arg == "--node-limit") {
            options.node_limit = std::stoull(require_value("--node-limit"));
        } else if (arg == "--no-resume") {
            options.resume = false;
        } else if (arg == "--self-test") {
            options.self_test = true;
        } else if (arg == "--combine") {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                options.combine_files.push_back(argv[++i]);
            }
            if (options.combine_files.empty()) {
                throw std::invalid_argument("--combine needs CSV files");
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: twoqubit_tstate_m" << M << " [options]\n\n"
                << "  --threads N             Parallel first-measurement jobs\n"
                << "  --q-start A             First unsigned Pauli label\n"
                << "  --q-end B               Last unsigned Pauli label\n"
                << "  --q-file FILE           Optional symmetry representatives\n"
                << "  --output FILE           CSV checkpoint/output\n"
                << "  --capacity-power P      Per-worker table has 2^P slots\n"
                << "  --progress-every N      Branch progress cadence (0 disables)\n"
                << "  --no-resume             Ignore existing CSV\n"
                << "  --self-test             Algebra and recursion validation\n"
                << "  --combine FILE...       Combine raw q-range CSVs\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown option: " + arg);
        }
    }

    if (options.threads <= 0) throw std::invalid_argument("threads must be positive");
    if (options.q_start < 1 || options.q_end >= UMAX ||
        options.q_start > options.q_end) {
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

        build_hypotheses();

        std::vector<int> q_values;
        if (!options.q_file.empty()) {
            q_values = read_q_file(options.q_file);
        } else {
            for (int q = options.q_start; q <= options.q_end; ++q) q_values.push_back(q);
        }

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

        std::cout << "problem = two-qubit binary discrimination with " << M
                  << " injected T states\n"
                  << "total qubits = " << N << "\n"
                  << "value denominator = " << VALUE_DEN << "\n"
                  << "requested first measurements = " << q_values.size() << "\n"
                  << "already completed = " << (q_values.size() - pending.size()) << "\n"
                  << "pending = " << pending.size() << "\n"
                  << "threads = " << options.threads << "\n"
                  << "flat memo capacity = 2^" << options.capacity_power << " slots\n"
                  << "estimated memo memory per worker = " << std::fixed
                  << std::setprecision(3) << gib_per_worker << " GiB\n"
                  << "one fixed-outcome branch has at most approximately "
                  << EXPECTED_BRANCH_PROJECTORS << " stabilizer projectors\n";

        if (pending.empty()) {
            std::unordered_map<int, RootResult> selected;
            for (int q : q_values) {
                const auto it = existing.find(q);
                if (it != existing.end()) selected.emplace(q, it->second);
            }
            print_summary(selected, q_values.size(), !options.q_file.empty());
            return 0;
        }

        std::ofstream output;
        {
            const bool file_exists = static_cast<bool>(std::ifstream(options.output));
            const auto mode = options.resume ? std::ios::app : std::ios::trunc;
            output.open(options.output, mode);
            if (!output) throw std::runtime_error("Could not open output file");
            if (!file_exists || !options.resume) {
                output << "# twoqubit_tstate_discrimination\n"
                       << "# tstates=" << M << "\n"
                       << "# total_qubits=" << N << "\n"
                       << "# value_den=" << VALUE_DEN << "\n"
                       << "# required_roots=" << q_values.size() << "\n"
                       << "# root_mode="
                       << (options.q_file.empty() ? "raw_all_paulis" : "symmetry_representatives")
                       << "\n"
                       << "q,a,b,plus_a,plus_b,minus_a,minus_b,nodes_plus,nodes_minus,seconds\n";
                output.flush();
            }
        }

        std::mutex result_mutex;
        std::atomic<size_t> next_index{0};
        std::atomic<size_t> finished{0};
        const auto all_started = std::chrono::steady_clock::now();

        auto worker = [&](int worker_id) {
            BranchSolver solver(options.capacity_power, options.progress_every,
                                options.node_limit, worker_id);
            while (true) {
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
                              << " dec=" << std::setprecision(12) << decv(result.value)
                              << " nodes=(" << result.nodes_plus << ','
                              << result.nodes_minus << ") seconds=" << result.seconds
                              << " total_elapsed=" << total_seconds << "s\n";
                }
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(options.threads);
        for (int t = 0; t < options.threads; ++t) threads.emplace_back(worker, t);
        for (auto& thread : threads) thread.join();

        std::unordered_map<int, RootResult> selected;
        for (int q : q_values) {
            const auto it = existing.find(q);
            if (it != existing.end()) selected.emplace(q, it->second);
        }
        print_summary(selected, q_values.size(), !options.q_file.empty());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
