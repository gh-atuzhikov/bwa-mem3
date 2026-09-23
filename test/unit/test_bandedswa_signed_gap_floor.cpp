// Regression for the 16-bit AVX2/AVX-512 gap-open floor.
// m11 may be negative after a mismatch, so unsigned-saturating subtraction
// does not implement max(m11 - gap_open, 0).

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "bandedSWA.h"

#if HAVE_BSW_VECTOR_8_16

namespace {

struct Out { int score, tle, gtle, qle, gscore, max_off; };
struct Fixture { std::string ref, query; int h0; };

int padded_pairs(int n) {
    return ((n + SIMD_WIDTH8 - 1) / SIMD_WIDTH8) * SIMD_WIDTH8 + MAX_LINE_LEN;
}

void build_matrix(int8_t mat[25]) {
    int k = 0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) mat[k++] = (i == j) ? 1 : -4;
        mat[k++] = -1;
    }
    for (int j = 0; j < 5; ++j) mat[k++] = -1;
}

std::vector<uint8_t> encode(const std::string &text) {
    std::vector<uint8_t> out(text.size());
    for (size_t i = 0; i < text.size(); ++i) out[i] = (uint8_t)(text[i] - '0');
    return out;
}

void check_batch(const std::vector<Fixture> &fixtures, int band) {
    const int stride = 128;
    int8_t mat[25];
    build_matrix(mat);
    BandedPairWiseSW bsw(6, 1, 6, 1, 100, 5, mat, 1, 4, 1);
    std::vector<uint8_t> refs((size_t)stride * fixtures.size() + MAX_LINE_LEN, 0);
    std::vector<uint8_t> queries((size_t)stride * fixtures.size() + MAX_LINE_LEN, 0);
    std::vector<SeqPair> pairs(padded_pairs((int)fixtures.size()));
    std::vector<Out> scalar(fixtures.size());
    for (size_t i = 0; i < fixtures.size(); ++i) {
        std::vector<uint8_t> ref = encode(fixtures[i].ref);
        std::vector<uint8_t> query = encode(fixtures[i].query);
        std::copy(ref.begin(), ref.end(), refs.begin() + i * stride);
        std::copy(query.begin(), query.end(), queries.begin() + i * stride);
        Out &out = scalar[i];
        out.score = bsw.scalarBandedSWA((int)query.size(), query.data(),
            (int)ref.size(), ref.data(), band, fixtures[i].h0,
            &out.qle, &out.tle, &out.gtle, &out.gscore, &out.max_off);
        SeqPair &pair = pairs[i];
        pair.id = (int)i; pair.idr = (int)(i * stride); pair.idq = (int)(i * stride);
        pair.len1 = (int)ref.size(); pair.len2 = (int)query.size(); pair.h0 = fixtures[i].h0;
        pair.seqid = pair.regid = 0; pair.tight_band = pair.chain_band = 0;
        pair.score = pair.tle = pair.gtle = pair.qle = pair.gscore = pair.max_off = -1;
    }
    bsw.getScores16(pairs.data(), refs.data(), queries.data(),
                    (int32_t)fixtures.size(), 1, band);
    for (size_t i = 0; i < fixtures.size(); ++i) {
        INFO("fixture=" << i << " len1=" << fixtures[i].ref.size()
             << " len2=" << fixtures[i].query.size());
        CHECK(pairs[i].score == scalar[i].score);
        CHECK(pairs[i].tle == scalar[i].tle);
        CHECK(pairs[i].gtle == scalar[i].gtle);
        CHECK(pairs[i].qle == scalar[i].qle);
        CHECK(pairs[i].gscore == scalar[i].gscore);
        CHECK(pairs[i].max_off == scalar[i].max_off);
    }
}

std::string repeat_codes(int length, int phase, bool add_n) {
    std::string value;
    value.reserve((size_t)length);
    for (int i = 0; i < length; ++i) {
        char base = (char)('0' + ((i + phase) % 4));
        if (add_n && i % 11 == 5) base = '4';
        value.push_back(base);
    }
    return value;
}

}  // namespace

TEST_CASE("bandedSWA 16-bit signed gap floor matches scalar on minimized production fixture"
          * doctest::test_suite("unit/bandedswa")) {
    const Fixture minimized = {
        "023100330101002323132", "233333333333333313333", 26
    };
    check_batch({minimized}, 19);
    Fixture no_n = minimized;
    for (char &base : no_n.ref) if (base == '4') base = '0';
    for (char &base : no_n.query) if (base == '4') base = '0';
    check_batch({minimized, no_n, minimized, no_n, minimized}, 19);
}

TEST_CASE("bandedSWA 16-bit signed gap floor is lane and boundary independent"
          * doctest::test_suite("unit/bandedswa")) {
    const int lengths[] = {15, 16, 17, 31, 32, 33, 63, 64, 65};
    std::vector<Fixture> fixtures;
    for (int length : lengths) {
        fixtures.push_back({repeat_codes(length, 0, false),
                            repeat_codes(length, 1, false), length % 9});
        fixtures.push_back({repeat_codes(length, 0, true),
                            repeat_codes(length, 2, true), 26});
    }
    check_batch(fixtures, 100);
}

TEST_CASE("bandedSWA 16-bit fixed-seed ACGTN repeat differential matches scalar"
          * doctest::test_suite("unit/bandedswa")) {
    std::mt19937 rng(375);
    std::vector<Fixture> fixtures;
    for (int i = 0; i < 256; ++i) {
        int length = 15 + (int)(rng() % 51);
        std::string ref = repeat_codes(length, (int)(rng() % 4), (rng() & 1) != 0);
        std::string query = repeat_codes(length, (int)(rng() % 4), (rng() & 1) != 0);
        for (int j = 0; j < length; ++j) {
            if (rng() % 19 == 0) query[j] = (char)('0' + (rng() % 5));
        }
        fixtures.push_back({ref, query, (int)(rng() % 40)});
    }
    check_batch(fixtures, 100);
}

#endif
