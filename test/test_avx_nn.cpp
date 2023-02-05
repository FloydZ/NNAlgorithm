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



TEST(NearestNeighbor2, Windowed) {
	constexpr static WindowedAVX2_Config config{256, 4, 50, 1u<<18, 12, 4, 0, 496};
	WindowedAVX2<config> algo{};
	algo.run();
	algo.bench();
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
