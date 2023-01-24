#ifndef NN_CODE_OPTIONS_H
#define NN_CODE_OPTIONS_H
#define TEST_BASE_LIST_SIZE_LOG 10
#define TEST_BASE_LIST_SIZE (1u << TEST_BASE_LIST_SIZE_LOG)
#define G_n 100
constexpr uint64_t w=1;
constexpr uint64_t N=100;
constexpr uint64_t r=3;
constexpr uint64_t d=8;
constexpr uint64_t epsilon=1;
constexpr uint64_t THRESHHOLD=40;
constexpr double ratio=0.5;
constexpr double gam=0.1*G_n;
#define ITERS  1
#define RUNS 1
//#define ALL_DELTA
#endif //NN_CODE_OPTIONS_H
