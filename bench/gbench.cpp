#include "benchmark/benchmark.h"
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>

#include "options.h"
#include "windowed_nn_v2.h"


NNList L1{1}, L2{1};
uint64_t pos1, pos2;


void setup(const benchmark::State& state) {
	NearestNeighbor::create_test_lists(L1, L2, TEST_BASE_LIST_SIZE, w, pos1, pos2);
}

void bench_quad() {

	const NNContainer gold1 = L1[pos1];
	const NNContainer gold2 = L2[pos2];
	NearestNeighbor nn{L1, L2, w, r, N, d};

	uint64_t found = nn.NN(gold1, gold2);
}

// Define another benchmark
static void BM_quad(benchmark::State& state) {
  std::string x = "hello";
  for (auto _ : state) {
    bench_quad();
  }
}


BENCHMARK(BM_quad)->Setup(setup);
BENCHMARK_MAIN();
