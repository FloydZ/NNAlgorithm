#include <gtest/gtest.h>
#include <cstdint>

#define NN_CONFIG_SET
#define TEST_BASE_LIST_SIZE (1u << 10u)

#include "helper.h"
#include "windowed_avx2.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

//TEST(NearestNeighbor2, Windowed) {
//	uint64_t pos1, pos2;
//	const uint64_t w = 0.2*G_n;       // omega
//	const uint64_t d = uint64_t(0.3*G_n);
//	const uint64_t r = 2;
//	const uint64_t N = 10000;
//	const uint64_t size = TEST_BASE_LIST_SIZE;
//	const bool find_all = true;
//	const uint64_t tresh = 100;
//
//	NNList L1{1}, L2{1};
//
//	NearestNeighbor::create_test_lists(L1, L2, size, w, pos1, pos2, true, r);
//	const NNContainer gold1 = L1[pos1];
//	const NNContainer gold2 = L2[pos2];
//
//	std::cout << "Solution should be: " << pos1 << " " << pos2 << "\n";
//	std::cout << "first: " << L1[pos1] << " w:" << L1[pos1].weight() << "\n";
//	std::cout << "second:" << L2[pos2] << " w:" << L2[pos2].weight() << "\n";
//	std::cout << "List Size: " << L1.size() << " " << L2.size() << "\n";
//	std::cout << "\n\n";
//
//	WindowedNearestNeighbor2 nn{L1, L2, w, r, N, d, find_all, tresh};
//	uint64_t found = nn.NN();
//	EXPECT_EQ((found > 0) || (!nn.sols_1.empty()), true);
//	EXPECT_EQ(nn.print_result(gold1, gold2), true);
//
//
//	// normal quadratic search
//	NearestNeighbor nnq{L1, L2, w, r, N, d};
//	nnq.NN();
//	EXPECT_EQ(nnq.print_result(gold1, gold2), true);
//}

TEST(NearestNeighbor2, Windowed) {
	WindowedAVX2 algo{};
	algo.run();
	algo.bench();
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
