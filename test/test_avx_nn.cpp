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

constexpr size_t LS = 7;// 1u << 10u;
constexpr static WindowedAVX2_Config global_config{256, 4, 20, LS, 22, 16, 0, 512};

TEST(PopCountAVX2, uint32_t) {
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

TEST(PopCountAVX2, uint64_t) {
	constexpr static WindowedAVX2_Config config{256, 4, 50, 1u<<8, 12, 4, 0, 496};
	WindowedAVX2<config> algo{};
	__m256i a = _mm256_setr_epi64x(0, 1, 2, 3);
	__m256i b = algo.popcount_avx2_64(a);

	U256i c = U256i {b};
	EXPECT_EQ(c.b[0], 0);
	EXPECT_EQ(c.b[1], 1);
	EXPECT_EQ(c.b[2], 1);
	EXPECT_EQ(c.b[3], 2);

	for (size_t i = 0; i < 1000000; i++) {
		const uint64_t a1 = fastrandombytes_uint64();
		const uint64_t a2 = fastrandombytes_uint64();
		const uint64_t a3 = fastrandombytes_uint64();
		const uint64_t a4 = fastrandombytes_uint64();

		a = _mm256_setr_epi64x(a1, a2, a3, a4);
		b = algo.popcount_avx2_64(a);

		c = U256i {b};
		EXPECT_EQ(c.b[0], __builtin_popcountll(a1));
		EXPECT_EQ(c.b[1], __builtin_popcountll(a2));
		EXPECT_EQ(c.b[2], __builtin_popcountll(a3));
		EXPECT_EQ(c.b[3], __builtin_popcountll(a4));
	}
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


	if constexpr (LS > 1u << 16) {
		algo.bruteforce_256(LS, LS);
		EXPECT_EQ(algo.solutions_nr, 1);
		EXPECT_EQ(algo.all_solutions_correct(), true);
	} else {
		for (int i = 0; i < 10000; ++i) {
			algo.bruteforce_256(LS, LS);
			EXPECT_EQ(algo.solutions_nr, 1);
			EXPECT_EQ(algo.all_solutions_correct(), true);
			algo.solutions_nr = 0;

			free(algo.L1);
			free(algo.L2);
			algo.generate_random_instance();
		}
	}
}

TEST(Bruteforce, avx_256) {
	constexpr static WindowedAVX2_Config config{256, 4, 1, LS, 80, 50, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();

	if constexpr (LS > 1u << 16) {
		algo.bruteforce_avx2_256(LS, LS);
		EXPECT_EQ(algo.solutions_nr, 1);
		EXPECT_EQ(algo.all_solutions_correct(), true);
	} else {
		for (int i = 0; i < 10000; ++i) {
			algo.bruteforce_avx2_256(LS, LS);
			EXPECT_EQ(algo.solutions_nr, 1);
			EXPECT_EQ(algo.all_solutions_correct(), true);
			algo.solutions_nr = 0;

			free(algo.L1);
			free(algo.L2);
			algo.generate_random_instance();
		}
	}
}

TEST(Bruteforce, avx2_256_ux4) {
	constexpr static WindowedAVX2_Config config{256, 4, 1, LS, 80, 20, 0, 512};
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
	constexpr static WindowedAVX2_Config config{256, 4, 1, LS, 25, 12, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();

	algo.bruteforce_avx2_256_32_8x8(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;
}

TEST(Bruteforce, avx2_256_64_4x4) {
	constexpr static WindowedAVX2_Config config{256, 4, 1, 32, 30, 16, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();

	if constexpr (LS > 1u << 16) {
		algo.bruteforce_avx2_256_64_4x4(LS, LS);
		EXPECT_EQ(algo.solutions_nr, 1);
		EXPECT_EQ(algo.all_solutions_correct(), true);
	} else {
		for (int i = 0; i < 10000; ++i) {
			algo.bruteforce_avx2_256_64_4x4(LS, LS);
			EXPECT_EQ(algo.solutions_nr, 1);
			EXPECT_EQ(algo.all_solutions_correct(), true);
			algo.solutions_nr = 0;

			free(algo.L1);
			free(algo.L2);
			algo.generate_random_instance();
		}
	}
}

TEST(NearestNeighborAVX, avx2_sort_nn_on64) {
	constexpr size_t LS = 1u << 18u;
	constexpr static WindowedAVX2_Config config{256, 4, 320, LS, 22, 16, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS*4*8);

	const uint64_t z = fastrandombytes_uint64();
	size_t e1 = algo.avx2_sort_nn_on64_simple<0>(LS, z, algo.L1);
	size_t e2 = algo.avx2_sort_nn_on64<0>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}

	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS*4*8);

	e1 = algo.avx2_sort_nn_on64_simple<1>(LS, z, algo.L1);
	e2 = algo.avx2_sort_nn_on64<1>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}

	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS*4*8);

	e1 = algo.avx2_sort_nn_on64_simple<2>(LS, z, algo.L1);
	e2 = algo.avx2_sort_nn_on64<2>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}


	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS*4*8);

	e1 =algo.avx2_sort_nn_on64_simple<3>(LS, z, algo.L1);
	e2 =algo.avx2_sort_nn_on64<3>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}
}

TEST(NearestNeighborAVX, MO1284Params_n256_r4) {
	constexpr size_t LS = 1u << 18u;
	constexpr static WindowedAVX2_Config config{256, 4, 300, LS, 23, 16, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();

	constexpr uint32_t nr_tries = 1;
	uint32_t sols= 0;
	for (size_t i = 0; i < nr_tries; i++) {
		algo.avx2_nn(LS, LS);
		sols += algo.solutions_nr;
		algo.solutions_nr = 0;

		free(algo.L1);
		free(algo.L2);
		algo.generate_random_instance();
	}

	EXPECT_EQ(sols, nr_tries);
}

TEST(NearestNeighborAVX, Dev) {
	constexpr static WindowedAVX2_Config config{256, 4, 500, 1u<<14, 22, 16, 0, 512};
	WindowedAVX2<config> algo{};
	algo.run();
	algo.bench();
}

int main(int argc, char **argv) {
	random_seed(time(NULL));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
