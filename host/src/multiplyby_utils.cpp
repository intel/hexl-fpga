#include "multiplyby_utils.h"
#include <NTL/ZZ.h>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

class Timer {
public:
    Timer(const std::string name, bool debug = false, size_t num_bytes = 0)
        : stopped_(false) {
        this->name_ = name;
        this->start_point_ = std::chrono::high_resolution_clock::now();
        this->num_bytes_ = num_bytes;
        this->debug_ = debug;
    }

    void stop() {
        auto end_point = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end_point - start_point_);
        if (debug_) {
            std::cout << name_ << " takes " << time_span.count() * 1000 << "ms";
            if (num_bytes_ != 0) {
                std::cout << ", num_bytes = " << num_bytes_ << ", throughput = "
                          << num_bytes_ / time_span.count() / 1024 / 1024
                          << "MB/s";
            }
            std::cout << std::endl;
        }
        stopped_ = true;
    }
    ~Timer() {
        if (!stopped_) stop();
    }

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_point_;
    bool stopped_;
    size_t num_bytes_;
    bool debug_;
};

static unsigned precompute_modulus_k(unsigned long modulus) {
    unsigned k;
    for (int i = 64; i > 0; i--) {
        if ((1UL << i) >= modulus) k = i;
    }

    return k;
}

static unsigned long precompute_modulus_r(unsigned long modulus) {
    __int128 a = 1;
    unsigned long k = precompute_modulus_k(modulus);
    unsigned long r = (a << (2 * k)) / modulus;
    return r;
}

static uint64_t get_y_barret(uint64_t y, uint64_t p) {
    __int128 a = y;
    a = a << 64;
    a = a / p;
    return (uint64_t)a;
}

static sycl::ulong2 mulmod_y_ext(uint64_t y, uint64_t p) {
    if (y == 0) return {0, 0};
    return {y, get_y_barret(y, p)};
}

static void PrintEventTime(sycl::event& e, const char* tag) {
    auto submit_time =
        e.get_profiling_info<sycl::info::event_profiling::command_submit>();
    auto start_time =
        e.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end_time =
        e.get_profiling_info<sycl::info::event_profiling::command_end>();
    std::cout << tag << " execution time: " << std::fixed
              << std::setprecision(3)
              << ((double)(end_time - start_time)) / 1000000.0
              << "ms, end time - submit time: "
              << ((double)(end_time - submit_time)) / 1000000.0
              << ", submit time = " << submit_time / 1000000
              << ", start time = " << start_time / 1000000
              << ", end_time time = " << end_time / 1000000 << " \n ";
}

void PreComputeBringToSetInternal(
    const std::vector<uint64_t>& pi, const std::vector<uint64_t>& qj,
    const std::vector<uint8_t>& qj_prime_index, uint64_t plainText,
    const std::vector<uint64_t>& special_primes, size_t& P, size_t& Q,
    size_t& I, std::vector<sycl::ulong2>& scale_param_set,
    bool added_primes_at_end, bool add_special_primes) {
    Timer timer("PreComputeScaleParamSet");
    std::vector<sycl::ulong2> pt;
    std::vector<sycl::ulong2> P_qj;
    std::vector<sycl::ulong2> pstar_inv;
    std::vector<sycl::ulong2> pstar_qj;

    // num of target primes that may add primes
    Q = qj.size();
    // num of drop primes
    P = 0;
    uint64_t t = plainText;

    // compute the num of drop primes
    for (auto prime : pi) {
        P += (std::find(qj.begin(), qj.end(), prime) == qj.end());
    }

    NTL::ZZ diffProd(1);
    for (int i = 0; i < P; i++) {
        diffProd *= pi[i];
    }

    // I is the added primes
    I = Q - (pi.size() - P);
    NTL::ZZ IProd(1);

    // configure the position of added primes
    if (added_primes_at_end) {
        for (int i = Q - I; i < Q; i++) {
            IProd *= qj[i];
        }
    } else {
        for (int i = 0; i < I; i++) {
            IProd *= qj[i];
        }
    }

    NTL::ZZ prodInv(NTL::InvMod(rem(diffProd, t), t));
    for (int i = 0; i < P; i++) {
        ulong pt_i = NTL::rem(prodInv * diffProd / pi[i], t);
        pt.push_back(mulmod_y_ext(pt_i, t));
        ulong p_star_inv_i =
            NTL::InvMod(NTL::rem(diffProd / pi[i], pi[i]), pi[i]);
        p_star_inv_i = NTL::rem(IProd * p_star_inv_i, pi[i]);
        pstar_inv.push_back(mulmod_y_ext(p_star_inv_i, pi[i]));
    }

    for (int i = 0; i < Q; i++) {
        ulong P_inv_qj_i = NTL::InvMod(NTL::rem(diffProd, qj[i]), qj[i]);

        for (int j = 0; j < P; j++) {
            pstar_qj.push_back(mulmod_y_ext(
                NTL::rem(diffProd / pi[j] * P_inv_qj_i, qj[i]), qj[i]));
        }
        P_inv_qj_i = NTL::rem(IProd * P_inv_qj_i, qj[i]);
        P_qj.push_back(mulmod_y_ext(P_inv_qj_i, qj[i]));
    }

    //  Prod of special primes mod qj
    if (add_special_primes) {
        NTL::ZZ prod_special_primes(1);
        for (int i = 0; i < special_primes.size(); i++) {
            prod_special_primes *= special_primes[i];
        }

        for (int i = 0; i < Q; i++) {
            P_qj.push_back(
                mulmod_y_ext(NTL::rem(prod_special_primes, qj[i]), qj[i]));
        }
    }

    // packing
    // pi - P
    for (size_t i = 0; i < P; i++) {
        double tmp = 1;
        tmp /= pi[i];
        scale_param_set.push_back({pi[i], *(ulong*)&tmp});
    }

    // qj - Q
    for (size_t i = 0; i < Q; i++) {
        scale_param_set.push_back(
            {qj[i], qj_prime_index.size() > 0 ? qj_prime_index[i] : 0});
    }

    // pt - P
    for (size_t i = 0; i < P; i++) {
        scale_param_set.push_back(pt[i]);
    }

    // pstar_inv - P
    for (size_t i = 0; i < P; i++) {
        scale_param_set.push_back(pstar_inv[i]);
    }

    // P_qj - Q
    for (size_t i = 0; i < Q; i++) {
        scale_param_set.push_back(P_qj[i]);
    }

    // R_qj - Q if applicable
    if (add_special_primes) {
        for (size_t i = 0; i < Q; i++) {
            scale_param_set.push_back(P_qj[Q + i]);
        }
    }

    // pstar_qj - Q*P
    for (size_t i = 0; i < Q; i++) {
        for (size_t j = 0; j < P; j++) {
            scale_param_set.push_back(pstar_qj[i * P + j]);
        }
    }
}

void PreComputeBringToSet(const std::vector<uint64_t>& all_primes,
                          const std::vector<uint8_t>& pi_primes_index,
                          const std::vector<uint8_t>& qj_prime_index,
                          std::vector<uint8_t>& pi_reorder_primes_index,
                          std::vector<sycl::ulong2>& scale_param_set, size_t& P,
                          size_t& Q, size_t& I, uint64_t plainText) {
    // the source primes
    std::vector<uint64_t> pi;
    // the target primes
    std::vector<uint64_t> qj;
    // compute the num of dropped primes
    int num_dropped_primes = 0;
    for (auto prime_index : pi_primes_index) {
        num_dropped_primes +=
            (std::find(qj_prime_index.begin(), qj_prime_index.end(),
                       prime_index) == qj_prime_index.end());
    }

    // reoder pi, put the dropped primes to the beginning
    for (size_t i = pi_primes_index.size() - num_dropped_primes;
         i < pi_primes_index.size(); i++) {
        pi_reorder_primes_index.push_back(pi_primes_index[i]);
        pi.push_back(all_primes[pi_primes_index[i]]);
    }
    for (size_t i = 0; i < pi_primes_index.size() - num_dropped_primes; i++) {
        pi_reorder_primes_index.push_back(pi_primes_index[i]);
        pi.push_back(all_primes[pi_primes_index[i]]);
    }

    // fill qj
    for (size_t i = 0; i < qj_prime_index.size(); i++) {
        qj.push_back(all_primes[qj_prime_index[i]]);
    }

    assert(pi_reorder_primes_index.size() == pi_primes_index.size());

    // double primes index
    auto len = pi_reorder_primes_index.size();
    for (size_t i = 0; i < len; i++) {
        pi_reorder_primes_index.push_back(pi_reorder_primes_index[i]);
    }

    std::vector<uint64_t> empty_vec;
    PreComputeBringToSetInternal(pi, qj, qj_prime_index, plainText, empty_vec,
                                 P, Q, I, scale_param_set, false, false);
}

void PreComputeBreakIntoDigits(
    const FpgaHEContext& context, const std::vector<uint8_t>& primes_index,
    std::vector<unsigned>& num_digits_primes,
    std::vector<sycl::ulong2>& packed_precomuted_params) {
    std::vector<sycl::ulong4> pstar_inv;
    std::vector<sycl::ulong2> pstar_qj;
    std::vector<sycl::ulong> P_qj;

    auto& all_primes = context.all_primes;
    auto& num_designed_digits_primes = context.num_digits_primes;
    unsigned num_special_primes = context.num_special_primes;

    std::vector<uint64_t> pi(primes_index.size());
    for (size_t i = 0; i < primes_index.size(); i++) {
        pi[i] = all_primes[primes_index[i]];
    }
    /* packed parameters */
    // FORMAT:
    // pi and pi recip - all normal primes and special primes
    // pstar_inv and pstar_inv_recip - all normal primes
    // P_qj - num_digit_primes (normal_primes/2) + special primes
    long num_normal_primes = pi.size() - num_special_primes;

    // compute the actual prime size of each digit
    // the last digit size maybe smaller than designed digit size
    int i = 0;
    int num_left_primes = num_normal_primes;
    while (num_left_primes > 0) {
        num_digits_primes.push_back(
            std::min(num_designed_digits_primes[i], (uint64_t)num_left_primes));
        num_left_primes -= num_designed_digits_primes[i];
        i++;
    }

    // compute the num of small primes
    long num_small_primes = 0;
    for (size_t i = 0; i < all_primes.size(); i++) {
        if (all_primes[i] == pi[0]) {
            // all the primes before the normal primes are small primes
            num_small_primes = i;
            break;
        }
    }

    // compute the offset of each digit
    std::vector<int> digits_offset;
    int lastDigitsOffset = num_small_primes;
    long num_digits = num_digits_primes.size();
    for (long i = 0; i < num_digits; i++) {
        digits_offset.push_back(lastDigitsOffset);
        lastDigitsOffset += num_digits_primes[i];
    }

    // compute the prod of each digit
    std::vector<NTL::ZZ> P;
    for (long i = 0; i < num_digits; i++) {
        NTL::ZZ tmp{1};
        for (int j = 0; j < num_digits_primes[i]; j++) {
            tmp *= all_primes[digits_offset[i] + j];
        }
        P.push_back(tmp);
    }

    // compute digitsQHatInv
    std::vector<NTL::ZZ> prodOfDesignedDigits;
    std::vector<NTL::ZZ> digitsQHatInv;

    for (long i = 0; i < num_digits; i++) {
        NTL::ZZ tmp{1};
        for (int j = 0; j < num_designed_digits_primes[i]; j++) {
            tmp *= all_primes[digits_offset[i] + j];
        }

        prodOfDesignedDigits.push_back(tmp);
    }

    // compute QHatInv
    for (long i = 0; i < num_digits; i++) {
        NTL::ZZ qhat{1};
        for (long j = 0; j < num_digits; j++) {
            if (j != i) {
                qhat *= prodOfDesignedDigits[j];
            }
        }
        auto qhat_inv = NTL::InvMod(qhat % prodOfDesignedDigits[i],
                                    prodOfDesignedDigits[i]);
        digitsQHatInv.push_back(qhat_inv);
    }

    // gererate the qj primes of each digits
    std::vector<uint64_t> digit_qj_primes[MAX_DIGITS];

    for (long j = 0; j < num_digits; j++) {
        for (long i = 0; i < pi.size(); i++) {
            if (i < digits_offset[j] ||
                i >= (digits_offset[j] + num_digits_primes[j]))
                digit_qj_primes[j].push_back(pi[i]);
        }
    }

    // pstar_inv has all the primes
    for (long j = 0; j < num_digits; j++) {
        for (long i = digits_offset[j];
             i < digits_offset[j] + num_digits_primes[j]; i++) {
            ulong p_star_inv_i =
                NTL::InvMod(NTL::rem(P[j] / pi[i], pi[i]), pi[i]);
            auto tmp = mulmod_y_ext(p_star_inv_i, pi[i]);
            auto tmp2 = mulmod_y_ext(NTL::InvMod(p_star_inv_i, pi[i]), pi[i]);

            pstar_inv.push_back({tmp.s0(), tmp.s1(), tmp2.s0(), tmp2.s1()});
        }
    }

    // std::cout << "pstar_inv.size() = " << pstar_inv.size() << std::endl;
    assert(num_normal_primes == pstar_inv.size());

    // compute pstar_qj
    for (int i = 0; i < MAX_SPECIAL_PRIMES + MAX_NORMAL_PRIMES / 2; i++) {
        for (long k = 0; k < num_digits; k++) {
            for (int j = 0; j < MAX_NORMAL_PRIMES / 2; j++) {
                // P* mod qj
                auto tmp =
                    i < digit_qj_primes[k].size() && j < num_digits_primes[k]
                        ? NTL::rem(P[k] / all_primes[digits_offset[k] + j],
                                   digit_qj_primes[k][i])
                        : 0;
                pstar_qj.push_back(mulmod_y_ext(tmp, digit_qj_primes[k][i]));
            }
        }
    }

    // comput P_qj
    for (long j = 0; j < num_digits; j++)
        for (int i = 0; i < pi.size(); i++) {
            P_qj.push_back(i < digit_qj_primes[j].size()
                               ? NTL::rem(P[j], digit_qj_primes[j][i])
                               : 0);
        }

    // compute pi_recip
    std::vector<ulong2> pi_with_recip;
    for (int i = 0; i < pi.size(); i++) {
        double pi_recip = (double)1 / pi[i];
        pi_with_recip.push_back({pi[i], *(ulong*)&pi_recip});
    }

    // packing now
    // pi and pi recip - all normal primes and special primes
    for (size_t i = 0; i < pi.size(); i++) {
        packed_precomuted_params.push_back(pi_with_recip[i]);
    }

    // pstar_inv and pstar_inv_recip - all normal primes
    for (size_t i = 0; i < pstar_inv.size(); i++) {
        auto tmp = pstar_inv[i];
        packed_precomuted_params.push_back({tmp.s0(), tmp.s1()});
    }

    // pstar_inv_recip - all normal primes
    for (size_t i = 0; i < pstar_inv.size(); i++) {
        auto tmp = pstar_inv[i];
        packed_precomuted_params.push_back({tmp.s2(), tmp.s3()});
    }

    // P_qj - pi.size() * 2
    for (size_t i = 0; i < P_qj.size(); i++) {
        packed_precomuted_params.push_back({P_qj[i], 0});
    }

    for (long j = 0; j < num_digits; j++) {
        for (int i = 0; i < num_digits_primes[j]; i++) {
            auto pi = all_primes[digits_offset[j] + i];
            long qhat_inv_pi = NTL::rem(digitsQHatInv[j], pi);
            packed_precomuted_params.push_back(mulmod_y_ext(qhat_inv_pi, pi));
        }
    }

    for (long k = 0; k < num_digits; k++) {
        for (int i = 0; i < MAX_SPECIAL_PRIMES + MAX_NORMAL_PRIMES; i++) {
            for (int j = 0; j < MAX_DIGIT_SIZE; j++) {
                if (i < digit_qj_primes[k].size() && j < num_digits_primes[k]) {
                    // P mod qj
                    auto tmp = NTL::rem(P[k] / all_primes[digits_offset[k] + j],
                                        digit_qj_primes[k][i]);
                    packed_precomuted_params.push_back(
                        mulmod_y_ext(tmp, digit_qj_primes[k][i]));
                } else {
                    packed_precomuted_params.push_back({0, 0});
                }
            }
        }
    }
}
void PreComputeTensorProduct(const FpgaHEContext& context,
                             const std::vector<uint8_t>& primes_index,
                             std::vector<ulong4>& params) {
    auto num_primes = primes_index.size();
    for (auto prime_index : primes_index) {
        auto prime = context.all_primes[prime_index];
        params.push_back({prime, precompute_modulus_r(prime),
                          precompute_modulus_k(prime), 0});
    }
}

void PreComputeKeySwitchDigits(const FpgaHEContext& context,
                               const std::vector<uint8_t>& primes_index,
                               std::vector<ulong4>& params) {
    // compute the diff value to prepare for the next prime
    // 0,0,0,1,1 -> 0,0,1,0,1
    // the last one doesn't matter
    auto num_primes = primes_index.size();
    std::vector<int> primes_index_offset(primes_index.size());
    for (size_t i = 0; i < primes_index.size(); i++) {
        primes_index_offset[i] = primes_index[i] - i;
    }
    for (size_t i = 0; i < primes_index.size() - 1; i++) {
        primes_index_offset[i] =
            primes_index_offset[i + 1] - primes_index_offset[i];
    }

    // compute P mod pi
    NTL::ZZ prod(1);
    for (int i = 0; i < context.num_special_primes; i++) {
        prod *= context.all_primes[context.all_primes.size() - 1 - i];
    }

    // pre-computing r and k for primes
    for (size_t i = 0; i < primes_index.size(); i++) {
        auto prime = context.all_primes[primes_index[i]];
        ulong4 tmp;
        uint64_t cur_primes_index_offset = primes_index_offset[i];
        tmp.s0() = prime | (cur_primes_index_offset << 60);
        tmp.s1() = NTL::rem(prod, prime);
        tmp.s2() = precompute_modulus_r(prime);
        tmp.s3() = precompute_modulus_k(prime);
        params.push_back(tmp);
    }
}