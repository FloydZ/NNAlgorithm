#include "benchmark/benchmark.h"
#include <cinttypes>
#include <cstdint>
#include <cstdlib>

#define ENABLE_BENCHMARK
#include "windowed_avx2.h"

constexpr size_t LS = 1u << 18u;
constexpr size_t d = 15;
constexpr size_t dk = 26;
constexpr static WindowedAVX2_Config config{256, 4, 200, LS, dk, d, 0, 512};
WindowedAVX2<config> algo{};

// Define another benchmark
static void BM_avx2_256_32_8x8(benchmark::State& state) {
	algo.generate_random_instance();
	for (auto _ : state) {
		algo.bruteforce_avx2_256_32_8x8(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}

static void BM_avx2_256_64_4x4(benchmark::State& state) {
	for (auto _ : state) {
		algo.bruteforce_avx2_256_64_4x4(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}


static void BM_NearestNeighborAVX(benchmark::State& state) {
	for (auto _ : state) {
		algo.bruteforce_avx2_256_64_4x4(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_avx2_256_32_8x8)->RangeMultiplier(2)->Range(128, 1u<<14)->Complexity();
BENCHMARK(BM_avx2_256_64_4x4)->RangeMultiplier(2)->Range(128, 1u<<14)->Complexity();
BENCHMARK(BM_NearestNeighborAVX)->RangeMultiplier(2)->Range(128, 1u<<14)->Complexity();

int main(int argc, char** argv) {
	random_seed(time(NULL));
	algo.generate_random_instance();

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}