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
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// Exact root-split Bellman solver for full discrimination of four
// equiprobable three-qubit stabilizer product states
//
//     |+10>, |0+1>, |10+>, |--->,
//
// assisted by M common |T> states, M=0,1,2,3.  Compile with
// -DTSTATES=M.  The allowed measurement protocols are arbitrary stabilizer
// operations: Clifford unitaries, stabilizer ancillas, adaptive Pauli
// measurements, and classical feed-forward.
//
// At a signed stabilizer-code branch S,
//
//   R(S) = max_i (1/4) Tr(Pi_S rho_i tensor T^M),
//   V(S) = max{R(S), max_q[V(S_{q,+})+V(S_{q,-})]}.
//
// The root is split over first Pauli measurements.  The supplied dependency-
// free Python exporter gives one representative for every orbit under a
// verified local-Clifford/qubit-permutation symmetry subgroup of the problem.

namespace {

constexpr int SYSTEM_QUBITS = 3;
#ifndef TSTATES
#define TSTATES 0
#endif
constexpr int M = TSTATES;
static_assert(M >= 0 && M <= 3, "TSTATES must be 0,1,2,3");
constexpr int N = SYSTEM_QUBITS + M;
constexpr int NUM_HYP = 4;
constexpr int NB = 2 * N;
constexpr int UMAX = 1 << NB;
constexpr int SLOTBITS = NB + 1;
constexpr int EXPECTATION_DEN = 1 << M;
constexpr int VALUE_DEN = EXPECTATION_DEN * NUM_HYP * (1 << N);
constexpr int DEFAULT_CAPACITY_POWER =
    (N == 6 ? 25 : (N == 5 ? 18 : (N == 4 ? 13 : 10)));
constexpr uint64_t EXPECTED_BRANCH_PROJECTORS =
    (N == 6 ? 21'555'667ULL :
     N == 5 ? 150'451ULL :
     N == 4 ? 2'467ULL : 91ULL);

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

// Local Hermitian Pauli labels I=0, X=1, Z=2, Y=3.
// OUT and PH encode A B = i^PH C.
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

int mulp(int n, int p, int q) {
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

std::vector<int> extend_code(const std::vector<int>& code, int q, int sign,
                             int n) {
    const int signed_q = q | (sign << (2 * n));
    std::vector<int> out;
    out.reserve(code.size() * 2);
    out.insert(out.end(), code.begin(), code.end());
    for (int g : code) out.push_back(mulp(n, g, signed_q));
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
    code = extend_code(code, d, sign, 3);  // signed coherent X^d generator
    for (int b : orth_basis3(d)) {
        code = extend_code(code, b << 3,
                           parity_u(static_cast<unsigned>(b & u)), 3);
    }
    if (code.size() != 8) throw std::logic_error("Invalid state stabilizer");
    return code;
}

std::array<std::array<Val, UMAX>, NUM_HYP> HYP{};

int local_expectation(char state_symbol, int local_pauli) {
    // Local Pauli labels: I=0, X=1, Z=2, Y=3.
    if (local_pauli == 0) return 1;
    if (state_symbol == '+' && local_pauli == 1) return 1;
    if (state_symbol == '-' && local_pauli == 1) return -1;
    if (state_symbol == '0' && local_pauli == 2) return 1;
    if (state_symbol == '1' && local_pauli == 2) return -1;
    return 0;
}

void build_hypotheses() {
    for (auto& row : HYP) for (auto& value : row) value = {0, 0};

    // Natural label order: |+10>, |0+1>, |10+>, |--->.
    const std::array<std::string, NUM_HYP> states = {
        "+10", "0+1", "10+", "---"
    };

    for (int label = 0; label < NUM_HYP; ++label) {
        for (int system_u = 0; system_u < 64; ++system_u) {
            const int sx = system_u & 7;
            const int sz = system_u >> 3;
            int system_value = 1;
            for (int q = 0; q < 3; ++q) {
                const int local = ((sx >> q) & 1) | (((sz >> q) & 1) << 1);
                system_value *= local_expectation(states[label][q], local);
            }
            if (system_value == 0) continue;

            int patterns = 1;
            for (int j = 0; j < M; ++j) patterns *= 3;  // I, X, Y.
            for (int pattern = 0; pattern < patterns; ++pattern) {
                int tmp = pattern;
                int x = sx, z = sz, weight = 0;
                for (int j = 0; j < M; ++j) {
                    const int choice = tmp % 3;
                    tmp /= 3;
                    if (choice == 0) continue;
                    ++weight;
                    x |= 1 << (3 + j);       // X or Y
                    if (choice == 2) z |= 1 << (3 + j);  // Y
                }

                Val coefficient;
                if ((weight & 1) == 0) {
                    const int r = weight / 2;
                    coefficient = {system_value * (1 << (M - r)), 0};
                } else {
                    const int r = (weight - 1) / 2;
                    coefficient = {0, system_value * (1 << (M - r - 1))};
                }
                HYP[label][x | (z << N)] = coefficient;
            }
        }
    }

    for (int i = 0; i < NUM_HYP; ++i) {
        if (!eqv(HYP[i][0], {EXPECTATION_DEN, 0})) {
            throw std::logic_error("Hypothesis identity expectation is not one");
        }
    }
}

struct Key {
    uint64_t lo = 0;
    uint64_t hi = 0;  // only low 14 bits are used for six rank rows
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
    return {static_cast<uint64_t>(packed),
            static_cast<uint64_t>(packed >> 64)};
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
            if ((rows[i] >> col) & 1) { pivot = i; break; }
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
    std::vector<int> code{0};
    for (int p : basis_from_key(key)) {
        code = extend_code(code, p & (UMAX - 1), (p >> NB) & 1, N);
    }
    return code;
}

bool insert_linear(std::array<int, NB>& basis, int v) {
    for (int bit = NB - 1; bit >= 0; --bit) {
        if (((v >> bit) & 1) == 0) continue;
        if (basis[bit] != 0) v ^= basis[bit];
        else { basis[bit] = v; return true; }
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
            if ((rows[i] >> col) & 1) { pivot = i; break; }
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
            if (parity_u(static_cast<unsigned>(rows[i] & v))) {
                v |= 1 << pivot_columns[i];
            }
        }
        result.push_back(v);
    }
    return result;
}

std::vector<int> logical_representatives(const Key& key) {
    const auto signed_basis = basis_from_key(key);
    const int mask = (1 << N) - 1;
    std::vector<int> stabilizer_unsigned;
    std::vector<int> constraints;
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
          mask_(capacity_ - 1), keys_(capacity_), values_(capacity_),
          stamps_(capacity_, 0) {
        if (capacity_power < 8 || capacity_power > 31) {
            throw std::invalid_argument("capacity power must be in [8,31]");
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
                "Flat memo exceeded 80% load; increase --capacity-power");
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
    double estimated_gib() const {
        const long double bytes = static_cast<long double>(capacity_) *
            (sizeof(Key) + sizeof(uint64_t) + sizeof(uint32_t));
        return static_cast<double>(bytes / (1024.0L * 1024.0L * 1024.0L));
    }

  private:
    size_t capacity_, mask_;
    std::vector<Key> keys_;
    std::vector<uint64_t> values_;
    std::vector<uint32_t> stamps_;
    uint32_t epoch_ = 1;
    size_t count_ = 0;
};

struct BranchStats {
    uint64_t completed = 0;
    double seconds = 0.0;
};

class BranchSolver {
  public:
    BranchSolver(int capacity_power, uint64_t progress_every,
                 uint64_t node_limit, int worker_id)
        : memo_(capacity_power), progress_every_(progress_every),
          node_limit_(node_limit), worker_id_(worker_id) {}

    Val solve_branch(int q, int outcome_sign, BranchStats& stats) {
        memo_.begin_epoch();
        completed_ = 0;
        branch_started_ = std::chrono::steady_clock::now();
        const std::vector<int> root_code{0};
        const auto child = extend_code(root_code, q, outcome_sign, N);
        const Val answer = solve(canonical_key(child));
        stats.completed = completed_;
        stats.seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - branch_started_).count();
        return answer;
    }

    double memo_gib() const { return memo_.estimated_gib(); }

    static std::mutex io_mutex;

  private:
    Val immediate_stop(const std::vector<int>& code, int rank,
                       Val* upper_out) const {
        std::array<Val, NUM_HYP> weights{};
        for (int p : code) {
            const int u = p & (UMAX - 1);
            const int sign = (p >> NB) & 1;
            for (int i = 0; i < NUM_HYP; ++i) {
                Val coefficient = HYP[i][u];
                if (sign) coefficient = negv(coefficient);
                weights[i] = addv(weights[i], coefficient);
            }
        }
        const int factor = 1 << (N - rank);
        Val best = {weights[0].a * factor, weights[0].b * factor};
        Val upper{};
        for (int i = 0; i < NUM_HYP; ++i) {
            const Val value = {weights[i].a * factor, weights[i].b * factor};
            if (lessv(best, value)) best = value;
            upper = addv(upper, value);
        }
        if (upper_out) *upper_out = upper;
        return best;
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
            for (int q : logical_representatives(key)) {
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
        if (progress_every_ != 0 && completed_ % progress_every_ == 0) {
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - branch_started_).count();
            std::lock_guard<std::mutex> lock(io_mutex);
            std::cerr << "worker=" << worker_id_
                      << " branch_completed=" << completed_
                      << " rank=" << rank << " memo=" << memo_.size()
                      << " nodeV=" << val_string(best)
                      << " dec=" << std::setprecision(10) << decv(best)
                      << " elapsed=" << seconds << "s\n";
        }
        return best;
    }

    FlatMemo memo_;
    uint64_t progress_every_, node_limit_, completed_ = 0;
    int worker_id_;
    std::chrono::steady_clock::time_point branch_started_;
};

std::mutex BranchSolver::io_mutex;

struct RootResult {
    int q = 0;
    Val value{}, plus{}, minus{};
    uint64_t nodes_plus = 0, nodes_minus = 0;
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
    std::string output = "four_state_tstate_root_results.csv";
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
                throw std::runtime_error("q-file contains invalid Pauli label");
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
        if (line.empty() || line[0] == '#' || line.rfind("q,", 0) == 0) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream parser(line);
        RootResult result;
        if (parser >> result.q >> result.value.a >> result.value.b
                   >> result.plus.a >> result.plus.b
                   >> result.minus.a >> result.minus.b
                   >> result.nodes_plus >> result.nodes_minus >> result.seconds) {
            results[result.q] = result;
        }
    }
    return results;
}

void append_result(std::ofstream& output, const RootResult& result) {
    output << result.q << ',' << result.value.a << ',' << result.value.b << ','
           << result.plus.a << ',' << result.plus.b << ','
           << result.minus.a << ',' << result.minus.b << ','
           << result.nodes_plus << ',' << result.nodes_minus << ','
           << std::setprecision(17) << result.seconds << '\n';
    output.flush();
}

void print_summary(const std::unordered_map<int, RootResult>& results,
                   size_t required_count, bool representatives_assumed) {
    // Stopping at the root gives the largest prior, 1/4.
    Val best = {VALUE_DEN / NUM_HYP, 0};
    int best_q = 0;
    for (const auto& [q, result] : results) {
        if (lessv(best, result.value)) { best = result.value; best_q = q; }
    }

    std::cout << "\n=== root summary ===\n"
              << "completed first measurements = " << results.size()
              << " / " << required_count << "\n"
              << "best q = " << best_q << "\n"
              << "best value = " << val_string(best)
              << " = " << std::setprecision(15) << decv(best) << "\n";
    if (results.size() == required_count) {
        std::cout << "RESULT_NUM " << best.a << ' ' << best.b
                  << " DEN " << VALUE_DEN << "\n";
        if (representatives_assumed) {
            std::cout << "Result is exact provided the q-file contains one "
                         "representative from every root-measurement orbit.\n";
        } else {
            std::cout << "Exact raw root search complete.\n";
        }
    } else {
        std::cout << "Partial run: the displayed value is only a lower bound.\n";
    }
}

void self_test() {
    build_hypotheses();
    for (int i = 0; i < NUM_HYP; ++i) {
        if (!eqv(HYP[i][0], {EXPECTATION_DEN, 0})) {
            throw std::logic_error("Identity expectation self-test failed");
        }
    }
    const std::vector<int> root{0};
    if (!is_zero_key(canonical_key(root))) {
        throw std::logic_error("Root key is not zero");
    }
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
        if (arg == "--threads") options.threads = std::stoi(require_value("--threads"));
        else if (arg == "--q-start") options.q_start = std::stoi(require_value("--q-start"));
        else if (arg == "--q-end") options.q_end = std::stoi(require_value("--q-end"));
        else if (arg == "--q-file") options.q_file = require_value("--q-file");
        else if (arg == "--output") options.output = require_value("--output");
        else if (arg == "--capacity-power") options.capacity_power = std::stoi(require_value("--capacity-power"));
        else if (arg == "--progress-every") options.progress_every = std::stoull(require_value("--progress-every"));
        else if (arg == "--node-limit") options.node_limit = std::stoull(require_value("--node-limit"));
        else if (arg == "--no-resume") options.resume = false;
        else if (arg == "--self-test") options.self_test = true;
        else if (arg == "--combine") {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                options.combine_files.push_back(argv[++i]);
            }
            if (options.combine_files.empty()) {
                throw std::invalid_argument("--combine needs CSV files");
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: four_state_tstate_discrimination [options]\n\n"
                << "  --threads N             parallel root jobs\n"
                << "  --q-file FILE           symmetry root representatives\n"
                << "  --q-start A --q-end B   raw root range\n"
                << "  --output FILE           append/resume CSV\n"
                << "  --capacity-power P      per-worker flat memo size 2^P\n"
                << "  --progress-every N      branch progress cadence\n"
                << "  --node-limit N          smoke-test branch limit\n"
                << "  --self-test             algebra/key/recursion smoke test\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown option: " + arg);
        }
    }
    if (options.threads <= 0) throw std::invalid_argument("threads must be positive");
    if (options.q_start < 1 || options.q_end >= UMAX || options.q_start > options.q_end) {
        throw std::invalid_argument("Invalid q range");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.self_test) { self_test(); return 0; }
        build_hypotheses();

        std::vector<int> q_values;
        if (!options.q_file.empty()) q_values = read_q_file(options.q_file);
        else for (int q = options.q_start; q <= options.q_end; ++q) q_values.push_back(q);

        auto existing = options.resume ? read_results_file(options.output)
                                       : std::unordered_map<int, RootResult>{};
        std::vector<int> pending;
        for (int q : q_values) if (existing.find(q) == existing.end()) pending.push_back(q);

        FlatMemo probe(options.capacity_power);
        const double gib_per_worker = probe.estimated_gib();
        std::cout << "problem = full discrimination of |+10>, |0+1>, |10+>, |--->\n"
                  << "hypotheses = 4, T states = " << M << ", total qubits = " << N << "\n"
                  << "value denominator = " << VALUE_DEN << "\n"
                  << "requested root measurements = " << q_values.size() << "\n"
                  << "already completed = " << (q_values.size() - pending.size()) << "\n"
                  << "pending = " << pending.size() << "\n"
                  << "threads = " << options.threads << "\n"
                  << "memo capacity = 2^" << options.capacity_power << " slots\n"
                  << "estimated memo memory per worker = " << std::fixed
                  << std::setprecision(2) << gib_per_worker << " GiB\n"
                  << "one fixed-outcome branch has at most about "
                  << EXPECTED_BRANCH_PROJECTORS << " projectors\n";

        std::ofstream output;
        if (!pending.empty()) {
            const bool file_exists = static_cast<bool>(std::ifstream(options.output));
            output.open(options.output, std::ios::app);
            if (!output) throw std::runtime_error("Could not open output file");
            if (!file_exists || !options.resume) {
                output << "# stabdisc four-state full discrimination with injected T states\n"
                       << "# TSTATES=" << M << " VALUE_DEN=" << VALUE_DEN << "\n"
                       << "q,a,b,plus_a,plus_b,minus_a,minus_b,"
                          "nodes_plus,nodes_minus,seconds\n";
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
                    std::chrono::steady_clock::now() - started).count();

                std::lock_guard<std::mutex> lock(result_mutex);
                existing[q] = result;
                append_result(output, result);
                const size_t done = ++finished;
                const double total_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - all_started).count();
                std::cout << "root_done=" << done << '/' << pending.size()
                          << " q=" << q << " candidate=" << val_string(result.value)
                          << " dec=" << std::setprecision(12) << decv(result.value)
                          << " nodes=(" << result.nodes_plus << ',' << result.nodes_minus
                          << ") seconds=" << result.seconds
                          << " total_elapsed=" << total_seconds << "s\n";
            }
        };

        std::vector<std::thread> threads;
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
