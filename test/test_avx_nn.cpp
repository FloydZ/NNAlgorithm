#include <gtest/gtest.h>
#include <cstdint>

#define NN_CONFIG_SET
#define TEST_BASE_LIST_SIZE (1u << 10u)

#include "helper.h"
#include "windowed_avx2.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

union U256i {
	__m256i v;
	uint32_t a[8];
	uint64_t b[4];
};

constexpr size_t LS = 1u << 10u;
constexpr static WindowedAVX2_Config global_config{256, 4, 20, 1u<<10, 22, 16, 0, 512};

TEST(AVX2, uint32_t) {
	constexpr static WindowedAVX2_Config config{256, 4, 50, 1u<<8, 12, 4, 0, 496};
	WindowedAVX2<config> algo{};
	__m256i a = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	//__m256i b = WindowedAVX2<config>::popcount_avx2_32(a);
	__m256i b = algo.popcount_avx2_32(a);

	U256i c = U256i {b};
	EXPECT_EQ(c.a[0], 0);
	EXPECT_EQ(c.a[1], 1);
	EXPECT_EQ(c.a[2], 1);
	EXPECT_EQ(c.a[3], 2);
	EXPECT_EQ(c.a[4], 1);
	EXPECT_EQ(c.a[5], 2);
	EXPECT_EQ(c.a[6], 2);
	EXPECT_EQ(c.a[7], 3);
}

TEST(Bruteforce, n32) {
	constexpr static WindowedAVX2_Config config{32, 1, 1, LS, 10, 5, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_32(LS, LS);
	EXPECT_GT(algo.solutions_nr, 0);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(Bruteforce, avx_32) {
	constexpr static WindowedAVX2_Config config{32, 1, 1, LS, 10, 5, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_avx2_32(LS, LS);
	EXPECT_GT(algo.solutions_nr, 0);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(Bruteforce, n64) {
	constexpr static WindowedAVX2_Config config{64, 1, 1, LS, 10, 5, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_64(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(Bruteforce, avx2_64) {
	constexpr static WindowedAVX2_Config config{64, 1, 1, LS, 10, 5, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_avx2_64(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(Bruteforce, avx2_64_1x1) {
	constexpr static WindowedAVX2_Config config{64, 1, 1, LS, 10, 5, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_avx2_64_1x1(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(Bruteforce, avx2_64_uxv) {
	constexpr static WindowedAVX2_Config config{64, 1, 1, LS, 10, 5, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_avx2_64_uxv<1,1>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_64_uxv<2,2>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_64_uxv<4,4>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_64_uxv<8,8>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_64_uxv<1,2>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_64_uxv<2,1>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_64_uxv<4,2>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

}

// TODO not working
//TEST(Bruteforce, avx2_64_uxv_shuffle) {
//	constexpr static WindowedAVX2_Config config{64, 1, 1, LS, 32, 17, 0, 512};
//	WindowedAVX2<config> algo{};
//	algo.generate_random_instance();
//
//	algo.bruteforce_avx2_64_uxv_shuffle<1,1>(LS, LS);
//	EXPECT_EQ(algo.solutions_nr, 1);
//	algo.solutions_nr = 0;
//
//	algo.bruteforce_avx2_64_uxv_shuffle<2,2>(LS, LS);
//	EXPECT_EQ(algo.solutions_nr, 1);
//	algo.solutions_nr = 0;
//
//	algo.bruteforce_avx2_64_uxv_shuffle<4,4>(LS, LS);
//	EXPECT_EQ(algo.solutions_nr, 1);
//	algo.solutions_nr = 0;
//
//	algo.bruteforce_avx2_64_uxv_shuffle<8,8>(LS, LS);
//	EXPECT_EQ(algo.solutions_nr, 1);
//	algo.solutions_nr = 0;
//
//	algo.bruteforce_avx2_64_uxv_shuffle<1,2>(LS, LS);
//	EXPECT_EQ(algo.solutions_nr, 1);
//	algo.solutions_nr = 0;
//
//	algo.bruteforce_avx2_64_uxv_shuffle<2,1>(LS, LS);
//	EXPECT_EQ(algo.solutions_nr, 1);
//	algo.solutions_nr = 0;
//
//	algo.bruteforce_avx2_64_uxv_shuffle<4,2>(LS, LS);
//	EXPECT_EQ(algo.solutions_nr, 1);
//  EXPECT_EQ(algo.all_solutions_correct(), true);
//	algo.solutions_nr = 0;
//
//}

TEST(Bruteforce, n128) {
	constexpr static WindowedAVX2_Config config{128, 1, 1, LS, 48, 32, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_128(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(Bruteforce, n256) {
	constexpr static WindowedAVX2_Config config{256, 4, 1, LS, 80, 50, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_256(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(Bruteforce, avx_256) {
	constexpr static WindowedAVX2_Config config{256, 4, 1, LS, 80, 50, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_avx2_256(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(Bruteforce, avx2_256_ux4) {
	constexpr static WindowedAVX2_Config config{256, 4, 1, LS, 80, 50, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();

	algo.bruteforce_avx2_256_ux4<1>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_256_ux4<2>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_256_ux4<4>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_256_ux4<8>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;
}


TEST(Bruteforce, avx2_256_32_ux8) {
	constexpr static WindowedAVX2_Config config{256, 4, 1, LS, 25, 4, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();

	algo.bruteforce_avx2_256_32_ux8<1>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_256_32_ux8<2>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_256_32_ux8<4>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_256_32_ux8<8>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;
}

TEST(Bruteforce, avx2_256_32_8x8) {
	constexpr static WindowedAVX2_Config config{256, 4, 1, LS, 25, 15, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();

	algo.bruteforce_avx2_256_32_8x8(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;
}

TEST(Bruteforce, avx2_256_64_4x4) {
	constexpr static WindowedAVX2_Config config{256, 4, 1, LS, 30, 25, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();

	algo.bruteforce_avx2_256_64_4x4(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;
}


TEST(NearestNeighborAVX, MO1284Params_n256_r4) {
	constexpr size_t LS = 1u << 14u;
	constexpr static WindowedAVX2_Config config{256, 4, 400, LS, 26, 18, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	algo.avx2_nn(LS, LS);

	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(NearestNeighborAVX, Dev) {
	constexpr static WindowedAVX2_Config config{256, 4, 400, 1u<<14, 22, 16, 0, 512};
	WindowedAVX2<config> algo{};
	algo.run();
	algo.bench();
}

int main(int argc, char **argv) {
	random_seed(time(NULL));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
