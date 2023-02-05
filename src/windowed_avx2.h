#ifndef NN_CODE_WINDOWED_AVX2_H
#define NN_CODE_WINDOWED_AVX2_H

#include <cstdint>
#include <cstdlib>
#include <vector>
#include <array>
#include <cassert>
#include <iostream>
#include "immintrin.h"

#include "random.h"

#define LOAD_ALIGNED
#ifdef LOAD_ALIGNED
#define LOAD256(x) _mm256_lddqu_si256(x)
#else
#define LOAD256(x) _mm256_load_si256(x)
#endif

#ifndef ASSERT
#define ASSERT(x) assert(x)
#endif

class WindowedAVX2_Config {
private:
	// disable the normal constructor
	WindowedAVX2_Config() : n(0), r(0), N(0), d(0), dk(0), LIST_SIZE(0), epsilon(0), BRUTEFORCE_THRESHOLD(0) {}
public:
	const uint32_t n, r, N, d, dk, LIST_SIZE, epsilon, BRUTEFORCE_THRESHOLD;
	constexpr WindowedAVX2_Config(const uint32_t n,
	                              const uint32_t r,
	                              const uint32_t N,
	                              const uint32_t ls,
	                              const uint32_t dk,
	                              const uint32_t d,
	                              const uint32_t epsilon,
	                              const uint32_t bf) :
	   n(n), r(r), N(N), d(d), dk(dk), LIST_SIZE(ls), epsilon(epsilon), BRUTEFORCE_THRESHOLD(bf)   {};
};

///
#define BRUTEFORCE256_32_8x8_STEP(stack, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8)    \
	stack[ 0] = (uint8_t)compare_256_32(a1, b1); \
	stack[ 1] = (uint8_t)compare_256_32(a1, b2); \
	stack[ 2] = (uint8_t)compare_256_32(a1, b3); \
	stack[ 3] = (uint8_t)compare_256_32(a1, b4); \
	stack[ 4] = (uint8_t)compare_256_32(a1, b5); \
	stack[ 5] = (uint8_t)compare_256_32(a1, b6); \
	stack[ 6] = (uint8_t)compare_256_32(a1, b7); \
	stack[ 7] = (uint8_t)compare_256_32(a1, b8); \
	stack[ 8] = (uint8_t)compare_256_32(a2, b1); \
	stack[ 9] = (uint8_t)compare_256_32(a2, b2); \
	stack[10] = (uint8_t)compare_256_32(a2, b3); \
	stack[11] = (uint8_t)compare_256_32(a2, b4); \
	stack[12] = (uint8_t)compare_256_32(a2, b5); \
	stack[13] = (uint8_t)compare_256_32(a2, b6); \
	stack[14] = (uint8_t)compare_256_32(a2, b7); \
	stack[15] = (uint8_t)compare_256_32(a2, b8); \
	stack[16] = (uint8_t)compare_256_32(a3, b1); \
	stack[17] = (uint8_t)compare_256_32(a3, b2); \
	stack[18] = (uint8_t)compare_256_32(a3, b3); \
	stack[19] = (uint8_t)compare_256_32(a3, b4); \
	stack[20] = (uint8_t)compare_256_32(a3, b5); \
	stack[21] = (uint8_t)compare_256_32(a3, b6); \
	stack[22] = (uint8_t)compare_256_32(a3, b7); \
	stack[23] = (uint8_t)compare_256_32(a3, b8); \
	stack[24] = (uint8_t)compare_256_32(a4, b1); \
	stack[25] = (uint8_t)compare_256_32(a4, b2); \
	stack[26] = (uint8_t)compare_256_32(a4, b3); \
	stack[27] = (uint8_t)compare_256_32(a4, b4); \
	stack[28] = (uint8_t)compare_256_32(a4, b5); \
	stack[29] = (uint8_t)compare_256_32(a4, b6); \
	stack[30] = (uint8_t)compare_256_32(a4, b7); \
	stack[31] = (uint8_t)compare_256_32(a4, b8); \
	stack[32] = (uint8_t)compare_256_32(a5, b1); \
	stack[33] = (uint8_t)compare_256_32(a5, b2); \
	stack[34] = (uint8_t)compare_256_32(a5, b3); \
	stack[35] = (uint8_t)compare_256_32(a5, b4); \
	stack[36] = (uint8_t)compare_256_32(a5, b5); \
	stack[37] = (uint8_t)compare_256_32(a5, b6); \
	stack[38] = (uint8_t)compare_256_32(a5, b7); \
	stack[39] = (uint8_t)compare_256_32(a6, b8); \
	stack[40] = (uint8_t)compare_256_32(a6, b1); \
	stack[41] = (uint8_t)compare_256_32(a6, b2); \
	stack[42] = (uint8_t)compare_256_32(a6, b3); \
	stack[43] = (uint8_t)compare_256_32(a6, b4); \
	stack[44] = (uint8_t)compare_256_32(a6, b5); \
	stack[45] = (uint8_t)compare_256_32(a6, b6); \
	stack[46] = (uint8_t)compare_256_32(a6, b7); \
	stack[47] = (uint8_t)compare_256_32(a6, b8); \
	stack[48] = (uint8_t)compare_256_32(a7, b1); \
	stack[49] = (uint8_t)compare_256_32(a7, b2); \
	stack[50] = (uint8_t)compare_256_32(a7, b3); \
	stack[51] = (uint8_t)compare_256_32(a7, b4); \
	stack[52] = (uint8_t)compare_256_32(a7, b5); \
	stack[53] = (uint8_t)compare_256_32(a7, b6); \
	stack[54] = (uint8_t)compare_256_32(a7, b7); \
	stack[55] = (uint8_t)compare_256_32(a8, b8); \
	stack[56] = (uint8_t)compare_256_32(a8, b1); \
	stack[57] = (uint8_t)compare_256_32(a8, b2); \
	stack[58] = (uint8_t)compare_256_32(a8, b3); \
	stack[59] = (uint8_t)compare_256_32(a8, b4); \
	stack[60] = (uint8_t)compare_256_32(a8, b5); \
	stack[61] = (uint8_t)compare_256_32(a8, b6); \
	stack[62] = (uint8_t)compare_256_32(a8, b7); \
	stack[63] = (uint8_t)compare_256_32(a8, b4)




// good values
//      n   N   r  k    dk   L     B
//      64  25  2  32   11  20    432
//      128 50  2  64   18  14    432 
//      256 50  4  64   25  18    108*4
//      256 10  8  32   12  18    108*8
template<const WindowedAVX2_Config &config>
class WindowedAVX2 {
public:
	//constexpr static size_t n = 256;         // n
	//constexpr static size_t r = 2;           // number of blocks
	//constexpr static size_t N = 25;          // number of list spawn
	//constexpr static size_t LIST_SIZE = 112;// well, list size
	////constexpr static size_t LIST_SIZE = 1u<<18u;// well, list size
	//constexpr static double d_ = 0.1;        // delta/n
	//constexpr static uint64_t k = n/r;       // n/r BlockSize
	//constexpr static uint32_t dk = 12;       // weight per block
	//constexpr static uint32_t d = 4;         // weight diff total to lok for
	//constexpr static uint64_t epsilon = 0;   // additional offset/variance we allow in each level to mach on.
	//constexpr static uint64_t BRUTEFORCE_THRESHHOLD = 8*104;

	constexpr static size_t n = config.n;
	constexpr static size_t r = config.r;
	constexpr static size_t N = config.N;
	constexpr static size_t LIST_SIZE = config.LIST_SIZE;
	constexpr static uint64_t k = n/r;
	constexpr static uint32_t dk = config.dk;
	constexpr static uint32_t d = config.d;
	constexpr static uint64_t epsilon = config.epsilon;
	constexpr static uint64_t BRUTEFORCE_THRESHHOLD = config.BRUTEFORCE_THRESHOLD;

	// only exact matching in the bruteforce step, if set to `true`
	constexpr static bool EXACT = false;

	// Array indicating the window boundaries.
	std::vector<uint64_t> buckets_windows{0, 32};

	/// Base types
	using T = uint64_t; // NOTE do not change.
	constexpr static size_t T_BITSIZE = sizeof(T) * 8;
	constexpr static size_t ELEMENT_NR_LIMBS = (n + T_BITSIZE - 1) / T_BITSIZE;
	using Element = T[ELEMENT_NR_LIMBS];

	// instance
	alignas(64) Element *L1, *L2;

	// solution
	size_t solution_l = 0, solution_r = 0, solutions_nr = 0;
	std::vector<std::pair<size_t, size_t>> solutions;

	/// choose a random element e1 s.t. \wt(e1) < d, and set e2 = 0
	/// \param e1
	/// \param e2
	static void generate_golden_element_simple(Element &e1, Element &e2) noexcept {
		constexpr T mask = n % T_BITSIZE == 0 ? T(-1) : ((1ul << n % T_BITSIZE) - 1ul);
		while (true) {
			uint32_t wt = 0;
			for (uint32_t i = 0; i < ELEMENT_NR_LIMBS - 1; i++) {
				e1[i] = fastrandombytes_uint64();
				e2[i] = 0;
				wt += __builtin_popcount(e1[i]);
			}

			e1[ELEMENT_NR_LIMBS - 1] = fastrandombytes_uint64() & mask;
			e2[ELEMENT_NR_LIMBS - 1] = 0;
			wt += __builtin_popcount(e1[ELEMENT_NR_LIMBS-1]);

			if (wt < d)
				return;
		}
	}

	/// chooses e1 completely random and e2 a weight d vector. finally e2 = e2 ^ e1
	/// \param e1 input/output
	/// \param e2 input/output
	static void generate_golden_element(Element &e1, Element &e2) noexcept {
		constexpr T mask = n % T_BITSIZE == 0 ? T(-1) : ((1ul << n % T_BITSIZE) - 1ul);
		static_assert(n > d);
		static_assert(64 > d);

		// clear e1, e2
		for (uint32_t i = 0; i < ELEMENT_NR_LIMBS; ++i) {
			e1[i] = 0;
			e2[i] = 0;
		}

		// choose e2;
		e2[0] = (1u << d) - 1;
		for (uint32_t i = 0; i < d; ++i) {
			const uint32_t pos = fastrandombytes_uint64() % (n - i - 1);

			const uint32_t from_limb = 0;
			const uint32_t from_pos = i;
			const T from_mask = 1u << from_pos;

			const uint32_t to_limb = pos / T_BITSIZE;
			const uint32_t to_pos = pos % T_BITSIZE;
			const T to_mask = 1u << to_pos;

			const T from_read = (e2[from_limb] & from_mask) >> from_pos;
			const T to_read = (e2[to_limb] & to_mask) >> to_pos;
			e2[to_limb] ^= (-from_read ^ e2[to_limb]) & (1ul << to_pos);
			e2[from_limb] ^= (-to_read ^ e2[from_limb]) & (1ul << from_pos);
		}


		uint32_t wt = 0;
		for (uint32_t i = 0; i < ELEMENT_NR_LIMBS - 1; i++) {
			e1[i] = fastrandombytes_uint64();
			e2[i] ^= e1[i];
			wt += __builtin_popcountll(e1[i] ^ e2[i]);
		}

		e1[ELEMENT_NR_LIMBS - 1] = fastrandombytes_uint64() & mask;
		e2[ELEMENT_NR_LIMBS - 1] ^= e1[ELEMENT_NR_LIMBS - 1];
		wt += __builtin_popcountll(e1[ELEMENT_NR_LIMBS-1] ^ e2[ELEMENT_NR_LIMBS-1]);
		ASSERT(wt = d);
	}

	///
	/// \param e
	static void generate_random_element(Element &e) noexcept {

		constexpr T mask = n%T_BITSIZE == 0 ? T(-1) : ((1ul << n%T_BITSIZE) - 1ul);
		for (uint32_t i = 0; i < ELEMENT_NR_LIMBS-1; i++) {
			e[i] = fastrandombytes_uint64();
		}

		e[ELEMENT_NR_LIMBS - 1] = fastrandombytes_uint64() & mask;
	}

	/// generate a random list
	/// \param L
	static void generate_random_lists(Element *L) noexcept {
		for (size_t i = 0; i < LIST_SIZE; i++) {
			generate_random_element(L[i]);
		}
	}

	/// generate a random instance, just for testing and debugging
	void generate_random_instance() noexcept {
		constexpr size_t list_size = (ELEMENT_NR_LIMBS * LIST_SIZE * sizeof(T));
		L1 = (Element *)aligned_alloc(64, list_size);
		L2 = (Element *)aligned_alloc(64, list_size);
		assert(L1);
		assert(L2);

		generate_random_lists(L1);
		generate_random_lists(L2);

		// generate solution:
		solution_l = 0;//fastrandombytes_uint64() % LIST_SIZE;
		solution_r = 0;//fastrandombytes_uint64() % LIST_SIZE;
		//std::cout << "sols at: " << solution_l << " " << solution_r << "\n";

		if constexpr (EXACT) {
			Element sol;
			generate_random_element(sol);

			// inject the solution
			for (uint32_t i = 0; i < ELEMENT_NR_LIMBS; ++i) {
				L1[solution_l][i] = sol[i];
				L2[solution_r][i] = sol[i];
			}
		} else {
			Element sol1, sol2;
			generate_golden_element(sol1, sol2);
			// inject the solution
			for (uint32_t i = 0; i < ELEMENT_NR_LIMBS; ++i) {
				L1[solution_l][i] = sol1[i];
				L2[solution_r][i] = sol2[i];
			}
		}


	}

	///
	/// \param mask
	/// \return
	__m256i bit_mask_64(const uint64_t mask) {
		ASSERT(mask < (1u<<8u));

		uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);
		expanded_mask *= 0xFFU;
		// the identity shuffle for vpermps, packed to one index per byte
		const uint64_t identity_indices = 0x0706050403020100;
		uint64_t wanted_indices = identity_indices & expanded_mask;
		const __m128i bytevec = _mm_cvtsi64_si128(wanted_indices);
		const __m256i shufmask = _mm256_cvtepu8_epi32(bytevec);
		return shufmask;
	}

	///
	/// \param wt
	void shuffle_2bit(const int wt){
		/// see the description of this magic in `shuffle_down_64`
		const uint64_t expanded_mask_org = _pdep_u64(wt, 0b0101010101010101);
		const uint64_t expanded_mask = expanded_mask_org * 0b11;
		constexpr uint64_t identity_indices = 0b11100100;
		constexpr uint64_t identity_mask = (1u << 8u) - 1u;

		const uint64_t wanted_indices_down1 = _pext_u64(identity_indices, expanded_mask & identity_mask);
		const uint64_t wanted_indices_up1   = _pdep_u64(identity_indices, expanded_mask & identity_mask);

		const uint64_t wanted_indices_down2 = _pext_u64(identity_indices, expanded_mask >> 8u);
		const uint64_t wanted_indices_up2 = _pdep_u64(identity_indices, expanded_mask >> 8u);
	}

	/// returns a permutation that shuffles down a mask on 32bit limbs
	/// e.g INPUT: 0b10000001
	///				<-    256    ->
	///    OUTPUT: [  0 ,..., 7, 0]
	///		   MSB <-32->
	/// 		   <-  8 limbs   ->
	/// to apply the resulting __m256i permutation use:
	///			const uint64_t shuffle = 0b1000001;
	/// 		const __m256i permuted_data = _mm256_permutevar8x32_ps(data, shuffle);
	/// \param mask bit mask. Must be smaller than 2**8
	/// \return the permutation
	__m256i shuffle_down_32(const uint64_t mask) const noexcept {
		// make sure only sane inputs make it.
		ASSERT(mask < (1u<<8u));

		uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);
		// mask |= mask<<1 | mask<<2 | ... | mask<<7;
		expanded_mask *= 0xFFU;
		// ABC... -> AAAAAAAABBBBBBBBCCCCCCCC...: replicate each bit to fill its byte

		// the identity shuffle for vpermps, packed to one index per byte
		const uint64_t identity_indices = 0x0706050403020100;
		uint64_t wanted_indices = _pext_u64(identity_indices, expanded_mask);

		const __m128i bytevec = _mm_cvtsi64_si128(wanted_indices);
		const __m256i shufmask = _mm256_cvtepu8_epi32(bytevec);
		return shufmask;
	}

	/// returns a permutation that shuffles down a mask on 32bit limbs
	/// e.g INPUT: 0b1001
	///				<-     256   ->
	///    OUTPUT: [  0  , 0, 3, 0]
	///		   MSB <-64->
	/// 		   <-  4 limbs   ->
	/// to apply the resulting __m256i permutation use:
	///			const uint64_t shuffle = 0b1000001;
	/// 		const __m256i permuted_data = _mm256_permutevar4x64_pd(data, shuffle);
	/// \param mask bit mask. Must be smaller than 2**4
	/// \return the permutation
	__m256i shuffle_down_64(const uint64_t mask) const noexcept {
		// make sure only sane inputs make it.
		ASSERT(mask < (1u<<4u));

		uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);  
		// mask |= mask<<1 | mask<<2 | ... | mask<<7;
		expanded_mask *= 0xFFU;  
		// ABC... -> AAAAAAAABBBBBBBBCCCCCCCC...: replicate each bit to fill its byte
		
		// the identity shuffle for vpermps, packed to one index per byte
		const uint64_t identity_indices = 0x0706050403020100;    
		uint64_t wanted_indices = _pext_u64(identity_indices, expanded_mask);

		const __m128i bytevec = _mm_cvtsi64_si128(wanted_indices);
		const __m256i shufmask = _mm256_cvtepu8_epi64(bytevec);
		return shufmask;
	}

	/// pretty much the same as `shuffle_down_64` but accepts permutation mask bigger than 2**4 up to
	/// 2**8, meaning this function returns 2 permutations for at most 2 * 4  uint64_t limbs.
	/// \param higher
	/// \param lower
	/// \param mask
	void shuffle_down_2_64(__m256i &higher, __m256i &lower, const uint64_t mask) const noexcept {
		// make sure only sane inputs make it.
		ASSERT(mask < (1u<<8u));

		/// see the description of this magic in `shuffle_down_64`
		uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);
		expanded_mask *= 0xFFU;
		const uint64_t identity_indices = 0x0302010003020100;
		uint64_t wanted_indices = _pext_u64(identity_indices, expanded_mask);

		const __m128i bytevec1 = _mm_cvtsi32_si128(uint16_t(wanted_indices));
		const __m128i bytevec2 = _mm_cvtsi32_si128(uint16_t(wanted_indices >> 16));
		lower = _mm256_cvtepu8_epi64(bytevec1);
		higher = _mm256_cvtepu8_epi64(bytevec2);
	}

	/// same as shuffle up, but instead a compressed array is expanded according to mask
	/// EXAMPLE: INPUT: 0b10100001
	/// 		<-      256        ->
	/// OUTPUT: [  2  , 0, 1, ..., 0]
	///			<-32->
	///			<-    8  limbs     ->
	/// USAGE:
	///			const uint64_t shuffle = 0b1000001;
	/// 		const __m256i permuted_data = _mm256_permutevar8x32_ps(data, shuffle);
	/// \param mask
	/// \return
	__m256i shuffle_up_32(const uint64_t mask) const noexcept {
		ASSERT(mask < (1u<<8u));

		uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);
		expanded_mask *= 0xFFU;
		const uint64_t identity_indices = 0x0706050403020100;
		uint64_t wanted_indices = _pdep_u64(identity_indices, expanded_mask);

		const __m128i bytevec = _mm_cvtsi64_si128(wanted_indices);
		const __m256i shufmask = _mm256_cvtepu8_epi32(bytevec);
		return shufmask;
	}

	/// same as shuffle up, but instead a compressed array is expanded according to mask
	/// EXAMPLE: INPUT: 0b1010
	/// 		<-     256    ->
	/// OUTPUT: [  1  , 0, 0, 0]
	///			<-64->
	///			<-   4 limbs  ->
	/// USAGE:
	///			const uint64_t shuffle = 0b1000001;
	/// 		const __m256i permuted_data = _mm256_permutevar4x64_pd(data, shuffle);
	/// \param mask
	/// \return
	__m256i shuffle_up_64(const uint64_t mask) const noexcept {
		ASSERT(mask < (1u<<4u));

		uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);
		expanded_mask *= 0xFFU;
		const uint64_t identity_indices = 0x03020100;
		uint64_t wanted_indices = _pdep_u64(identity_indices, expanded_mask);

		const __m128i bytevec = _mm_cvtsi64_si128(wanted_indices);
		const __m256i shufmask = _mm256_cvtepu8_epi64(bytevec);
		return shufmask;
	}

	/// similar to `shuffle_up_64`, but instead it can shuffle up to 8 64bit
	///	limbs in parallel. Therefore it needs to return 2 __m256i
	/// \param mask
	void shuffle_up_2_64(__m256i &higher, __m256i &lower, const uint64_t mask) const noexcept {
		ASSERT(mask < (1u<<8u));

		uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);
		expanded_mask *= 0xFFU;
		const uint64_t identity_indices = 0x03020100;
		uint64_t wanted_indices1 = _pdep_u64(identity_indices, expanded_mask & ((1ul << 32u) - 1));
		uint64_t wanted_indices2 = _pdep_u64(identity_indices, expanded_mask >> 32u);

		const __m128i bytevec1 = _mm_cvtsi32_si128(wanted_indices1);
		const __m128i bytevec2 = _mm_cvtsi32_si128(wanted_indices2);
		lower = _mm256_cvtepu8_epi64(bytevec1);
		higher = _mm256_cvtepu8_epi64(bytevec2);
	}

	//constexpr static uint8_t popcount_table[256] = {0 ,1 ,1 ,2 ,1 ,2 ,2 ,3 ,1 ,2 ,2 ,3 ,2 ,3 ,3 ,4 ,1 ,2 ,2 ,3 ,2 ,3 ,3 ,4 ,2 ,3 ,3 ,4 ,3 ,4 ,4 ,5 ,1 ,2 ,2 ,3 ,2 ,3 ,3 ,4 ,2 ,3 ,3 ,4 ,3 ,4 ,4 ,5 ,2 ,3 ,3 ,4 ,3 ,4 ,4 ,5 ,3 ,4 ,4 ,5 ,4 ,5 ,5 ,6 ,1 ,2 ,2 ,3 ,2 ,3 ,3 ,4 ,2 ,3 ,3 ,4 ,3 ,4 ,4 ,5 ,2 ,3 ,3 ,4 ,3 ,4 ,4 ,5 ,3 ,4 ,4 ,5 ,4 ,5 ,5 ,6 ,2 ,3 ,3 ,4 ,3 ,4 ,4 ,5 ,3 ,4 ,4 ,5 ,4 ,5 ,5 ,6 ,3 ,4 ,4 ,5 ,4 ,5 ,5 ,6 ,4 ,5 ,5 ,6 ,5 ,6 ,6 ,7 ,1 ,2 ,2 ,3 ,2 ,3 ,3 ,4 ,2 ,3 ,3 ,4 ,3 ,4 ,4 ,5 ,2 ,3 ,3 ,4 ,3 ,4 ,4 ,5 ,3 ,4 ,4 ,5 ,4 ,5 ,5 ,6 ,2 ,3 ,3 ,4 ,3 ,4 ,4 ,5 ,3 ,4 ,4 ,5 ,4 ,5 ,5 ,6 ,3 ,4 ,4 ,5 ,4 ,5 ,5 ,6 ,4 ,5 ,5 ,6 ,5 ,6 ,6 ,7 ,2 ,3 ,3 ,4 ,3 ,4 ,4 ,5 ,3 ,4 ,4 ,5 ,4 ,5 ,5 ,6 ,3 ,4 ,4 ,5 ,4 ,5 ,5 ,6 ,4 ,5 ,5 ,6 ,5 ,6 ,6 ,7 ,3 ,4 ,4 ,5 ,4 ,5 ,5 ,6 ,4 ,5 ,5 ,6 ,5 ,6 ,6 ,7 ,4 ,5 ,5 ,6 ,5 ,6 ,6 ,7 ,5 ,6 ,6 ,7 ,6 ,7 ,7 };
	//inline static __m256i popcount_avx2_32_v2(const __m256i vec) noexcept {
	//	const static __m256i low_mask = _mm256_set1_epi16(0xff);
	//	const __m256i lo  = _mm256_and_si256(vec, low_mask);
	//	const __m256i hi  = _mm256_and_si256(_mm256_srli_epi16(vec, 8), low_mask);
	//	const __m256i pop1 = _mm256_i32gather_epi32((uint32_t *)popcount_table, lo, 1);
	//	const __m256i pop2 = _mm256_i32gather_epi32((uint32_t *)popcount_table, hi, 1);
	//	return _mm256_and_si256(_mm256_add_epi8(pop1, pop2), low_mask);
	//}

	const __m256i lookup = _mm256_setr_epi8(
			/* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
			/* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
			/* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
			/* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,
			/* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
			/* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
			/* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
			/* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4
	);
	/// special popcount, which popcounts on 8 * 32u bit limbs in parallel
	inline __m256i popcount_avx2_32(const __m256i vec) noexcept {

		const __m256i low_mask = _mm256_set1_epi8(0x0f);
	    const __m256i lo  = _mm256_and_si256(vec, low_mask);
	    const __m256i hi  = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask);
	    const __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
	    const __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
	    __m256i local = _mm256_setzero_si256();
	    local = _mm256_add_epi8(local, popcnt1);
	    local = _mm256_add_epi8(local, popcnt2);
	
		// not the best
		const __m256i mask = _mm256_set1_epi32(0xff);
		__m256i ret = _mm256_and_si256(local, mask);
		ret = _mm256_add_epi8(ret, _mm256_and_si256(_mm256_srli_epi16(local,  8), mask));
		ret = _mm256_add_epi8(ret, _mm256_and_si256(_mm256_srli_epi32(local, 16), mask));
		//ret = _mm256_add_epi8(ret, _mm256_and_si256(_mm256_srli_epi32(local, 24), mask));
		return ret;
	}
	
	/// special popcount which popcounts on 4 * 64 bit limbs in parallel
	inline static __m256i popcount_avx2_64(const __m256i vec) noexcept {
		const __m256i lookup = _mm256_setr_epi8(
		    /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
		    /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
		    /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
		    /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,
		    /* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
		    /* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
		    /* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
		    /* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4
		);
	
		const __m256i low_mask = _mm256_set1_epi8(0x0f);
	    const __m256i lo  = _mm256_and_si256(vec, low_mask);
	    const __m256i hi  = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask);
	    const __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
	    const __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
	    __m256i local = _mm256_setzero_si256();
	    local = _mm256_add_epi8(local, popcnt1);
	    local = _mm256_add_epi8(local, popcnt2);
	
		const __m256i mask2 = _mm256_set1_epi64x(0xff);
		__m256i ret;
		
		ret = _mm256_add_epi8(local, _mm256_srli_epi32(local,  8));
		ret = _mm256_add_epi8(ret, _mm256_srli_epi32(ret,  16));
		ret = _mm256_add_epi8(ret, _mm256_srli_epi64(ret,  32));
		ret = _mm256_and_si256(ret, mask2);
		return ret;
	}

	// adds `li` and `lr` to the solutions list.
	void found_solution(const size_t li,
	                    const size_t lr) {
		ASSERT(li < LIST_SIZE);
		ASSERT(lr < LIST_SIZE);
		solutions.resize(solutions_nr + 1);
		solutions[solutions_nr++] = std::pair<size_t, size_t>{li, lr};
	}

	// checks whether all submitted solutions are correct
	bool all_solutions_correct() {
		if (solutions_nr == 0)
			return false;
		
		for (uint32_t i = 0; i < solutions_nr; i++) {
			bool equal = true;
			if constexpr (EXACT) {
				for (uint32_t j = 0; j < ELEMENT_NR_LIMBS; j++) {
					equal &= L1[solutions[i].first][j] == L2[solutions[i].second][j];
				}
			} else {
				uint32_t wt = 0;
				for (uint32_t j = 0; j < ELEMENT_NR_LIMBS; j++) {
					wt += __builtin_popcountll(L1[solutions[i].first][j] ^ L2[solutions[i].second][j]);
				}

				equal = wt <= d;
			}

			if (!equal)
				return false;
		}

		return true;
	}

	/// checks whether a,b are a solution or not
	/// NOTE: upper bound `d` is inclusive
	inline bool compare_u32(const uint32_t a, const uint32_t b) {
		if constexpr(EXACT) {
			return a == b;
		} else {
			return uint32_t(__builtin_popcount(a ^ b)) <= d;
		}
	}

	/// checks whether a,b are a solution or not
	/// NOTE: upper bound `d` is inclusive
	/// \param a
	/// \param b
	/// \return
	inline bool compare_u64(const uint64_t a, const uint64_t b) {
		if constexpr(EXACT) {
			return a == b;
		} else {
			return uint32_t(__builtin_popcountll(a ^ b)) <= d;
		}
	}

	inline bool compare_u64_ptr(const uint64_t *a, const uint64_t *b) {
		if constexpr(EXACT) {
			for (uint32_t i = 0; i < ELEMENT_NR_LIMBS; i++) {
				if (a[i] != b[i])
					return false;
			}

			return true;
		} else {
			uint32_t wt = 0;
			for (uint32_t i = 0; i < ELEMENT_NR_LIMBS; i++) {
				wt += __builtin_popcountll(a[i] ^ b[i]);
			}

			return wt <= d;
		}
	}

	///
	/// NOTE: upper bound `d` is inclusive
	/// \param in
	/// \return
	inline int compare_256_32(const __m256i in1, const __m256i in2) {
		if constexpr(EXACT) {
			const __m256i tmp2 = _mm256_cmpeq_epi32(in1, in2);
			return _mm256_movemask_ps((__m256) tmp2);
		} else {
			const static __m256i weight = _mm256_set1_epi32(d+1);

			const __m256i tmp1 = _mm256_xor_si256(in1, in2);
			const __m256i pop = popcount_avx2_32(tmp1);
			const __m256i tmp2 = _mm256_cmpgt_epi32(weight, pop);
			return _mm256_movemask_ps((__m256) tmp2);
		}
	}

	///
	/// NOTE: upper bound `d` is inclusive
	/// \param in
	/// \return
	inline int compare_256_64(const __m256i in1, const __m256i in2) {
		if constexpr(EXACT) {
			const __m256i tmp2 = _mm256_cmpeq_epi64(in1, in2);
			return _mm256_movemask_pd((__m256) tmp2);
		} else {
			const static __m256i weight = _mm256_set1_epi64x(d+1);

			const __m256i tmp1 = _mm256_xor_si256(in1, in2);
			const __m256i pop = popcount_avx2_64(tmp1);
			const __m256i tmp2 = _mm256_cmpgt_epi64(weight, pop);
			return _mm256_movemask_pd((__m256) tmp2);
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: without avx2
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_32(const size_t e1,
	                   const size_t e2) noexcept {
		ASSERT(n <= 32);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// limb position to compare on, basically the column to compare on.
		constexpr uint32_t limb_pos = 0;

		for (size_t i = s1; i < e1; ++i) {
			for (size_t j = s2; j < s2+e2; ++j) {
				if (compare_u32(L1[i][limb_pos], L2[j][limb_pos])) {
					found_solution(i, j);
				}
			}
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: only compares a single 32 bit column of the list. But its
	///			still possible
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 23...43) is impossible.
	/// NOTE: uses avx2
	/// NOTE: only a single 32bit element is compared.
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_avx2_32(const size_t e1,
	                        const size_t e2) noexcept {
		ASSERT(n <= 32);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// limb position to compare on, basically the column to compare on.
		constexpr uint32_t limb_pos = 0;

		/// strive is the number of bytes to the next element
		constexpr uint32_t stride = 8;

		/// difference of the memory location in the right list
		const __m256i loadr = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

		/// NOTE: only possible because L2 is a continuous memory block
		const T *ptr_r = (T *)L2;

		for (size_t i = s1; i < e1; ++i) {
			// NOTE: implicit typecast because T = uint64
			const __m256i li = _mm256_set1_epi32(L1[i][limb_pos]);

			for (size_t j = s2; j < s2+(e2+7)/8; ++j) {
				const __m256i ri = _mm256_i32gather_epi32(ptr_r + 8*j, loadr, stride);
				const int m = compare_256_32(li, ri);

				if (m) {
					const size_t jprime = j*8 + __builtin_ctz(m);
					//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
					found_solution(i, jprime);
				}
			}
		}
	}


	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: without avx2
	/// \param s1 start index of list 1
	/// \param e1 end index of list 1
	/// \param s2 start index list 2
	/// \param e2 end index list 2
	void bruteforce_64(const size_t e1,
	                   const size_t e2) noexcept {
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// limb position to compare on, basically the column to compare on.
		constexpr uint32_t limb_pos = 0;

		for (size_t i = s1; i < e1; ++i) {
			for (size_t j = s2; j < s2+e2; ++j) {
				if (L1[i][limb_pos] == L2[j][limb_pos]) {
					found_solution(i, j);
				}
			}
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: without avx2
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_128(const size_t e1,
						const size_t e2) {		
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		for (size_t i = s1; i < e1; i++) {
			for (size_t j = s2; j < e2; j++) {
				uint32_t weight = 0;
	
				#pragma unroll
				for (uint32_t s = 0; s < ELEMENT_NR_LIMBS; s++){
					weight += L1[i][s] != L2[j][s];
				}
	
				if (weight == 0) {
					found_solution(i, j);
				}
			}
		}
	}


	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: without avx2
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_256(const size_t e1,
	                    const size_t e2) {
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		for (size_t i = s1; i < e1; i++) {
			for (size_t j = s2; j < e2; j++) {
				uint32_t weight = 0;

				#pragma unroll
				for (uint32_t s = 0; s < ELEMENT_NR_LIMBS; s++){
					weight += L1[i][s] != L2[j][s];
				}

				if (weight == 0) {
					found_solution(i, j);
				}
			}
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: uses avx2
	/// NOTE: only compares a single 64 bit column of the list. But its
	///			still possible
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_avx2_64(const size_t e1,
	                        const size_t e2) noexcept {
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// limb position to compare on, basically the column to compare on.
		constexpr uint32_t limb_pos = 0;

		/// strive is the number of bytes to the next element
		constexpr uint32_t stride = 8;

		/// difference of the memory location in the right list
		const __m128i loadr = {(1ull << 32u),  (2ul) | (3ull << 32u)};

		/// allowed weight to match on
		const __m256i weight = _mm256_setr_epi64x(0, 0, 0, 0);

		for (size_t i = s1; i < e1; ++i) {
			const __m256i li = _mm256_set1_epi64x(L1[i][limb_pos]);

			/// NOTE: only possible because L2 is a continuous memory block
			T *ptr_r = (T *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; ++j, ptr_r += 4) {
				const __m256i ri = _mm256_i32gather_epi64(ptr_r, loadr, stride);
				const int m = compare_256_64(li, ri);

				if (m) {
					const size_t jprime = j*4+__builtin_ctz(m);
					//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
					found_solution(i, jprime);
				}
			}
		}
	}

	/// NOTE: in comparison to the other version `bruteforce_avx2_64` this implementation
	///			assumes that the elements to compare are fully compared on all n variables
	///  		e.g. ELEMENT_NR_LIMBS == 1
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_avx2_64_1x1(const size_t e1,
	                            const size_t e2) noexcept {
		ASSERT(ELEMENT_NR_LIMBS == 1);
		ASSERT(n <= 64);
		ASSERT(n >= 33);

		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// allowed weight to match on
		const __m256i weight = _mm256_setr_epi64x(0, 0, 0, 0);

		for (size_t i = s1; i < e1; ++i) {
			const __m256i li = _mm256_set1_epi64x(L1[i][0]);

			/// NOTE: only possible because L2 is a continuous memory block
			__m256i *ptr_r = (__m256i *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; ++j, ptr_r += 1) {
				const __m256i ri = LOAD256(ptr_r);
				const int m = compare_256_64(li, ri);

				if (m) {
					const size_t jprime = j*4 + __builtin_ctz(m);
					// std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
					found_solution(i, jprime);
				}
			}
		}
	}

	/// NOTE: in comparison to the other version `bruteforce_avx2_64` this implementation
	///			assumes that the elements to compare are fully compared on all n variables
	///  		e.g. ELEMENT_NR_LIMBS == 1
	/// NOTE: compared to `bruteforce_avx2_64_1x1` this unrolls `u` elementes in the left
	///			list and `v` elements on the right.
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	template<const uint32_t u, const uint32_t v>
	void bruteforce_avx2_64_uxv(const size_t e1,
	                            const size_t e2) noexcept {
		ASSERT(ELEMENT_NR_LIMBS == 1);
		ASSERT(n <= 64);
		ASSERT(n >= 33);

		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		__m256i lii[u], rii[v];

		/// allowed weight to match on
		const __m256i weight = _mm256_setr_epi64x(0, 0, 0, 0);

		for (size_t i = s1; i < e1; i += u) {

			#pragma unroll
			for (uint32_t j = 0; j < u; ++j) {
				lii[j] = _mm256_set1_epi64x(L1[i + j][0]);
			}

			/// NOTE: only possible because L2 is a continuous memory block
			__m256i *ptr_r = (__m256i *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; j += v, ptr_r += v) {

				#pragma unroll
				for (uint32_t s = 0; s < v; ++s) {
					rii[s] = LOAD256(ptr_r + s);
				}

				#pragma unroll
				for (uint32_t a1 = 0; a1 < u; ++a1) {
					const __m256i tmp1 = lii[a1];
					#pragma unroll
					for (uint32_t a2 = 0; a2 < v; ++a2) {
						const __m256i tmp2 = rii[a2];
						const int m = compare_256_64(tmp1, tmp2);

						if (m) {
							const size_t jprime = j*4 + a2*4 + __builtin_ctz(m);
							const size_t iprime = i + a1;
							//std::cout << L1[iprime][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
							found_solution(iprime, jprime);
						}
					}
				}
			}
		}
	}

	/// NOTE: in comparison to the other version `bruteforce_avx2_64` this implementation
	///			assumes that the elements to compare are fully compared on all n variables
	///  		e.g. ELEMENT_NR_LIMBS == 1
	/// NOTE: compared to `bruteforce_avx2_64_1x1` this unrolls `u` elementes in the left
	///			list and `v` elements on the right.
	/// NOTE: compared to `bruteforce_avx2_64_uxv` this function is not only comparing 1
	///			element of the left list with u elements from the right. Side
	///			Internally the loop is unrolled to compare u*4 elements to v on the right
	/// NOTE: assumes the input list to of length multiple of 16
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	template<const uint32_t u, const uint32_t v>
	void bruteforce_avx2_64_uxv_shuffle(const size_t e1,
	                                    const size_t e2) noexcept {
		ASSERT(ELEMENT_NR_LIMBS == 1);
		ASSERT(n <= 64);
		ASSERT(n >= 33);

		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		__m256i lii[u], rii[v];
		__m256i *ptr_l = (__m256i *)L1;

		/// allowed weight to match on
		const __m256i weight = _mm256_setr_epi64x(0, 0, 0, 0);

		for (size_t i = s1; i < s1 + (e1+3)/4; i += u, ptr_l += u) {

			#pragma unroll
			for (uint32_t j = 0; j < u; ++j) {
				lii[j] = LOAD256(ptr_l + j);
			}

			/// NOTE: only possible because L2 is a continuous memory block
			__m256i *ptr_r = (__m256i *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; j += v, ptr_r += v) {

				#pragma unroll
				for (uint32_t s = 0; s < v; ++s) {
					rii[s] = LOAD256(ptr_r + s);
				}

				#pragma unroll
				for (uint32_t a1 = 0; a1 < u; ++a1) {
					const __m256i tmp1 = lii[a1];
					#pragma unroll
					for (uint32_t a2 = 0; a2 < v; ++a2) {
						__m256i tmp2 = rii[a2];
						int m = compare_256_64(tmp1, tmp2);
						if (m) {
							const size_t jprime = j*4 + a2*4 + __builtin_ctz(m);
							const size_t iprime = i*4 + a1*4 + __builtin_ctz(m);
							//std::cout << L1[iprime][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
							found_solution(iprime, jprime);
						}

						tmp2 = _mm256_permute4x64_epi64(tmp2, 0b10010011);
						m = compare_256_64(tmp1, tmp2);
						if (m) {
							const size_t jprime = j*4 + a2*4 + __builtin_ctz(m) + 3;
							const size_t iprime = i*4 + a1*4 + __builtin_ctz(m);
							//std::cout << L1[iprime][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
							found_solution(iprime, jprime);
						}

						tmp2 = _mm256_permute4x64_epi64(tmp2, 0b10010011);
						m = compare_256_64(tmp1, tmp2);
						if (m) {
							const size_t jprime = j*4 + a2*4 + __builtin_ctz(m) + 2;
							const size_t iprime = i*4 + a1*4+ __builtin_ctz(m);
							//std::cout << L1[iprime][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
							found_solution(iprime, jprime);
						}

						tmp2 = _mm256_permute4x64_epi64(tmp2, 0b10010011);
						m = compare_256_64(tmp1, tmp2);
						if (m) {
							const size_t jprime = j*4 + a2*4 + __builtin_ctz(m) + 1;
							const size_t iprime = i*4 + a1*4 + __builtin_ctz(m);
							//std::cout << L1[iprime][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
							found_solution(iprime, jprime);
						}
					}
				}
			}
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: uses avx2
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_avx2_128(const size_t e1,
	                         const size_t e2) noexcept {

		ASSERT(n <= 128);
		ASSERT(n > 64);
		ASSERT(2 == ELEMENT_NR_LIMBS);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// difference of the memory location in the right list
		const __m128i loadr1 = {       (2ull << 32u), (4ul) | (6ull << 32u)};
		const __m128i loadr2 = {1ull | (3ull << 32u), (5ul) | (7ull << 32u)};

		/// allowed weight to match on
		const __m256i weight = _mm256_setr_epi64x(0, 0, 0, 0);

		for (size_t i = s1; i < e1; ++i) {
			const __m256i li1 = _mm256_set1_epi64x(L1[i][0]);
			const __m256i li2 = _mm256_set1_epi64x(L1[i][1]);

			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			T *ptr_r = (T *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; ++j, ptr_r += 8) {
				const __m256i ri = _mm256_i32gather_epi64(ptr_r, loadr1, 8);
				const int m1 = compare_256_64(li1, ri);

				if (m1) {
					const __m256i ri = _mm256_i32gather_epi64(ptr_r, loadr2, 8);
					const int m1 = compare_256_64(li2, ri);

					if (m1) {
						const size_t jprime = j*4;
						//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
						found_solution(i, jprime + __builtin_ctz(m1));
					}
				}
			}
		}
	}


	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: uses avx2
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// NOTE: assumes that list size is multiple of 4.
	/// NOTE: this check every element on the left against 4 on the right
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_avx2_256(const size_t e1,
	                         const size_t e2) noexcept {

		ASSERT(n <= 256);
		ASSERT(n > 128);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// difference of the memory location in the right list
		const __m128i loadr1 = {       (4ull << 32u), ( 8ul) | (12ull << 32u)};
		const __m128i loadr2 = {1ull | (5ull << 32u), ( 9ul) | (13ull << 32u)};
		const __m128i loadr3 = {2ull | (6ull << 32u), (10ul) | (14ull << 32u)};
		const __m128i loadr4 = {3ull | (7ull << 32u), (11ul) | (15ull << 32u)};

		/// allowed weight to match on
		const __m256i weight = _mm256_setr_epi64x(0, 0, 0, 0);

		for (size_t i = s1; i < e1; ++i) {
			const __m256i li1 = _mm256_set1_epi64x(L1[i][0]);
			const __m256i li2 = _mm256_set1_epi64x(L1[i][1]);
			const __m256i li3 = _mm256_set1_epi64x(L1[i][2]);
			const __m256i li4 = _mm256_set1_epi64x(L1[i][3]);

			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			T *ptr_r = (T *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; ++j, ptr_r += 16) {
				const __m256i ri = _mm256_i32gather_epi64(ptr_r, loadr1, 8);
				const int m1 = compare_256_64(li1, ri);

				if (m1) {
					const __m256i ri = _mm256_i32gather_epi64(ptr_r, loadr2, 8);
					const int m1 = compare_256_64(li2, ri);

					if (m1) {
						const __m256i ri = _mm256_i32gather_epi64(ptr_r, loadr3, 8);
						const int m1 = compare_256_64(li3, ri);

						if (m1) {
							const __m256i ri = _mm256_i32gather_epi64(ptr_r, loadr4, 8);
							const int m1 = compare_256_64(li4, ri);
							if (m1) {
								const size_t jprime = j*4;
								//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
								found_solution(i, jprime + __builtin_ctz(m1));
							}
						}
					}
				}
			}
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: uses avx2
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// NOTE: assumes that list size is multiple of 4.
	/// NOTE: additional to 'bruteforce_avx2_256' this functions unrolls `u` elements of
	///		the left list.
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	template<uint32_t u>
	void bruteforce_avx2_256_ux4(const size_t e1,
	                             const size_t e2) noexcept {
		static_assert(u > 0, "");
		static_assert(u <= 8, "");

		ASSERT(n <= 256);
		ASSERT(n > 128);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// difference of the memory location in the right list
		const __m128i loadr1 = {       (4ull << 32u), ( 8ul) | (12ull << 32u)};
		const __m128i loadr2 = {1ull | (5ull << 32u), ( 9ul) | (13ull << 32u)};
		const __m128i loadr3 = {2ull | (6ull << 32u), (10ul) | (14ull << 32u)};
		const __m128i loadr4 = {3ull | (7ull << 32u), (11ul) | (15ull << 32u)};

		alignas(32) __m256i li[u*4u];
		alignas(32) uint32_t m1s[8] = {0}; // this clearing is important

		/// allowed weight to match on
		uint32_t m1s_tmp = 0;
		const uint32_t m1s_mask = 1u << 31;

		for (size_t i = s1; i < s1 + e1; i += u) {
			/// Example u = 2
			/// li[0] = L[0][0]
			/// li[1] = L[1][0]
			/// li[2] = L[0][1]
			/// ...
			#pragma unroll
			for (uint32_t ui = 0; ui < u; ui++) {
				li[ui + 0*u] = _mm256_set1_epi64x(L1[i + ui][0]);
				li[ui + 1*u] = _mm256_set1_epi64x(L1[i + ui][1]);
				li[ui + 2*u] = _mm256_set1_epi64x(L1[i + ui][2]);
				li[ui + 3*u] = _mm256_set1_epi64x(L1[i + ui][3]);
			}


			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			T *ptr_r = (T *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; ++j, ptr_r += 16) {
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi64(ptr_r, loadr1, 8);
					const uint32_t tmp  = compare_256_64(li[0 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256d) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi64(ptr_r, loadr2, 8);
					const uint32_t tmp  = compare_256_64(li[1 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256d) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi64(ptr_r, loadr3, 8);
					const uint32_t tmp  = compare_256_64(li[2 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256d) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi64(ptr_r, loadr4, 8);
					const uint32_t tmp  = compare_256_64(li[3 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256d) LOAD256((__m256i *)m1s));
				if (m1s_tmp) {
					const size_t jprime = j*4;
					const uint32_t m1s_ctz = __builtin_ctz(m1s_tmp);
					const uint32_t bla = __builtin_ctz(m1s[m1s_ctz]);
					const size_t iprime = i + m1s_ctz;
					//std::cout << L1[iprime][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
					found_solution(iprime, jprime + bla);
				}
			}
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: uses avx2
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// NOTE: assumes that list size is multiple of 8.
	/// NOTE: additional to 'bruteforce_avx2_256' this functions unrolls `u` elements of
	///		the left list.
	///	NOTE: compared to `bruteforce_avx2_256_ux4` this function compares on 32 bit
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	template<uint32_t u>
	void bruteforce_avx2_256_32_ux8(const size_t e1,
	                                const size_t e2) noexcept {
		static_assert(u > 0, "");
		static_assert(u <= 8, "");

		ASSERT(n <= 256);
		ASSERT(n > 128);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// difference of the memory location in the right list
		const __m256i loadr1 = _mm256_setr_epi32(0, 8, 16, 24, 32, 40, 48, 56);
		const __m256i loadr_add = _mm256_set1_epi32(1);
		__m256i loadr;

		alignas(32) __m256i li[u*8u];
		alignas(32) uint32_t m1s[8] = {0}; // this clearing is important

		/// allowed weight to match on
		const __m256i weight = _mm256_set1_epi32(0);
		uint32_t m1s_tmp = 0;
		const uint32_t m1s_mask = 1u << 31;

		for (size_t i = s1; i < s1 + (e1+u-1); i += u) {
			/// Example u = 2
			/// li[0] = L[0][0]
			/// li[1] = L[1][0]
			/// li[2] = L[0][1]
			/// ...
			#pragma unroll
			for (uint32_t ui = 0; ui < u; ui++) {
				#pragma unroll
				for (uint32_t uii = 0; uii < 8; uii++) {
					li[ui + uii*u] = _mm256_set1_epi32(((uint32_t *)L1[i + ui])[uii]);
				}
			}


			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			uint32_t *ptr_r = (uint32_t *)L2;

			for (size_t j = s2; j < s2+(e2+7)/8; ++j, ptr_r += 64) {
				loadr = loadr1;
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32(ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[0 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256d) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = _mm256_add_epi32(loadr, loadr_add);
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32(ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[1 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256d) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = _mm256_add_epi32(loadr, loadr_add);
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32(ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[2 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256d) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = _mm256_add_epi32(loadr, loadr_add);
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32(ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[3 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256d) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = _mm256_add_epi32(loadr, loadr_add);
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32(ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[4 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256d) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = _mm256_add_epi32(loadr, loadr_add);
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32(ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[5 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256d) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = _mm256_add_epi32(loadr, loadr_add);
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32(ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[6 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256d) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = _mm256_add_epi32(loadr, loadr_add);
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32(ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[7 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256d) LOAD256((__m256i *)m1s));
				if (m1s_tmp) {
					const size_t jprime = j*8;
					const uint32_t m1s_ctz = __builtin_ctz(m1s_tmp);
					const uint32_t bla = __builtin_ctz(m1s[m1s_ctz]);
					const size_t iprime = i + m1s_ctz;
					//std::cout << L1[iprime][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
					found_solution(iprime, jprime + bla);
				}
			}
		}
	}

	template<const uint32_t off>
	void bruteforce_avx2_256_32_8x8_helper(uint32_t m1sx,
	                                       const uint8_t *m1s,
	                                       const uint32_t *ptr_l,
	                                       const uint32_t *ptr_r,
	                                       const size_t i, const size_t j) noexcept {
		while (m1sx > 0) {
			const uint32_t ctz = off + __builtin_ctz(m1sx);
			const uint32_t m1sc = m1s[ctz];
			const uint32_t m1sc_ctz = __builtin_ctz(m1sc);

			const uint32_t test_j = ctz % 8;
			const uint32_t test_i = ctz / 8;

			// TODO wahrscheinlich auch nicht richtig
			const uint32_t off_l = test_i * 8 + m1sc_ctz + 0;
			const uint32_t off_r = test_j * 8 + m1sc_ctz + 0;

			const uint64_t *test_tl = (uint64_t *) (ptr_l + off_l);
			const uint64_t *test_tr = (uint64_t *) (ptr_l + off_r);
			if (compare_u64_ptr(test_tl, test_tr)) {
				found_solution(i + off_l, j + off_r);
			}
			
			// TODO also not correct
			m1sx = 0;
			//m1sx ^= 1u << ctz;
		}
	}

	/// NOTE: this is hyper optimized for the case if there is only one solution.
	void bruteforce_avx2_256_32_8x8(const size_t e1,
	                                const size_t e2) noexcept {
		ASSERT(n <= 256);
		ASSERT(n > 128);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		uint32_t *ptr_l = (uint32_t *)L1;

		/// difference of the memory location in the right list
		const __m256i loadr1 = _mm256_setr_epi32(0, 8, 16, 24, 32, 40, 48, 56);

		alignas(32) uint8_t m1s[64];

		/// allowed weight to match on
		const __m256i zero = _mm256_set1_epi32(0);

		for (size_t i = s1; i < s1 + e1; i += 64, ptr_l += 512) {
			const __m256i l1 = _mm256_i32gather_epi32(ptr_l +   0, loadr1, 4);
			const __m256i l2 = _mm256_i32gather_epi32(ptr_l +  64, loadr1, 4);
			const __m256i l3 = _mm256_i32gather_epi32(ptr_l + 128, loadr1, 4);
			const __m256i l4 = _mm256_i32gather_epi32(ptr_l + 192, loadr1, 4);
			const __m256i l5 = _mm256_i32gather_epi32(ptr_l + 256, loadr1, 4);
			const __m256i l6 = _mm256_i32gather_epi32(ptr_l + 320, loadr1, 4);
			const __m256i l7 = _mm256_i32gather_epi32(ptr_l + 384, loadr1, 4);
			const __m256i l8 = _mm256_i32gather_epi32(ptr_l + 448, loadr1, 4);

			uint32_t *ptr_r = (uint32_t *)L2;
			for (size_t j = s1; j < s2 + e2; j += 64, ptr_r += 512) {
				const __m256i r1 = _mm256_i32gather_epi32(ptr_r +   0, loadr1, 4);
				const __m256i r2 = _mm256_i32gather_epi32(ptr_r +  64, loadr1, 4);
				const __m256i r3 = _mm256_i32gather_epi32(ptr_r + 128, loadr1, 4);
				const __m256i r4 = _mm256_i32gather_epi32(ptr_r + 192, loadr1, 4);
				const __m256i r5 = _mm256_i32gather_epi32(ptr_r + 256, loadr1, 4);
				const __m256i r6 = _mm256_i32gather_epi32(ptr_r + 320, loadr1, 4);
				const __m256i r7 = _mm256_i32gather_epi32(ptr_r + 384, loadr1, 4);
				const __m256i r8 = _mm256_i32gather_epi32(ptr_r + 448, loadr1, 4);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				uint32_t m1s1 = _mm256_movemask_epi8(_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s +  0)), zero));
				uint32_t m1s2 = _mm256_movemask_epi8(_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s + 32)), zero));
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r2, r3, r4, r5, r6, r7, r8, r1);
				m1s1 = _mm256_movemask_epi8(_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s +  0)), zero));
				m1s2 = _mm256_movemask_epi8(_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s + 32)), zero));
				// TODO from here on not correct rotation not included.
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r3, r4, r5, r6, r7, r8, r1, r2);
				m1s1 = _mm256_movemask_epi8(_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s +  0)), zero));
				m1s2 = _mm256_movemask_epi8(_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s + 32)), zero));
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r4, r5, r6, r7, r8, r1, r2, r3);
				m1s1 = _mm256_movemask_epi8(_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s +  0)), zero));
				m1s2 = _mm256_movemask_epi8(_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s + 32)), zero));
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r5, r6, r7, r8, r1, r2, r3, r4);
				m1s1 = _mm256_movemask_epi8(_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s +  0)), zero));
				m1s2 = _mm256_movemask_epi8(_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s + 32)), zero));
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r6, r7, r8, r1, r2, r3, r4, r5);
				m1s1 = _mm256_movemask_epi8(_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s +  0)), zero));
				m1s2 = _mm256_movemask_epi8(_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s + 32)), zero));
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r7, r8, r1, r2, r3, r4, r5, r6);
				m1s1 = _mm256_movemask_epi8(_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s +  0)), zero));
				m1s2 = _mm256_movemask_epi8(_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s + 32)), zero));
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r8, r1, r2, r3, r4, r5, r6, r7);
				m1s1 = _mm256_movemask_epi8(_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s +  0)), zero));
				m1s2 = _mm256_movemask_epi8(_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s + 32)), zero));
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32>(m1s2, m1s, ptr_l, ptr_r, i, j); }
			}
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: uses avx2
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// NOTE: assumes that list size is multiple of 4.
	/// NOTE: in comparison to `bruteforce_avx_256` this implementation used no `gather`
	///		instruction, but rather direct (aligned) loads.
	/// NOTE: only works if exact matching is active
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_avx2_256_v2(const size_t e1,
	                            const size_t e2) noexcept {
		ASSERT(EXACT);
		ASSERT(n <= 256);
		ASSERT(n > 128);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// allowed weight to match on
		const __m256i weight = _mm256_setr_epi64x(0, 0, 0, 0);
		__m256i *ptr_l = (__m256i *)L1;

		for (size_t i = s1; i < e1; ++i, ptr_l += 1) {
			const __m256i li1 = LOAD256(ptr_l);

			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			__m256i *ptr_r = (__m256i *)L2;

			for (size_t j = s2; j < s2+e2; ++j, ptr_r += 1) {
				const __m256i ri = LOAD256(ptr_r);
				const __m256i tmp1 = _mm256_xor_si256(li1, ri);
				const __m256i tmp2 = _mm256_cmpeq_epi64(tmp1, weight);
				const int m1 = _mm256_movemask_pd((__m256d) tmp2);

				if (m1) {
					//std::cout << L1[i][0] << " " << L2[j][0] << " " << L2[j+1][0] << " " << L2[j-1][0] << "\n";
					found_solution(i, j);
				}
			}
		}
	}


	/// NOTE: uses avx2 instructions
	/// NOTE: currently not correct
	/// \param gt_mask comparison mask. From this mask we can deduct the
	/// \param ctr
	/// \return
	size_t swap_avx_64(const __m256i gt_mask, const size_t ctr, Element *ptr, Element *L) {
		ASSERT(n <= 64);
		ASSERT(n > 32);

		// extract the int bit mask
		const int wt = _mm256_movemask_ps((__m256) gt_mask);

		// make sure only sane inputs make it.
		ASSERT(wt < (1u<<8u));

		/// see the description of this magic in `shuffle_down_64`
		const uint64_t expanded_mask_org = _pdep_u64(wt, 0b0101010101010101);
		const uint64_t expanded_mask = expanded_mask_org * 0b11;
		constexpr uint64_t identity_indices = 0b11100100;
		constexpr uint64_t identity_mask = (1u << 8u) - 1u;

		if (wt & 0b1111) {
			const uint64_t indices_down = _pext_u64(identity_indices, expanded_mask & identity_mask);
			const uint64_t indices_up   = _pdep_u64(identity_indices, expanded_mask & identity_mask);

			//const __m256i lower_down = _mm256_cvtepu8_epi64(_mm_cvtsi32_si128(uint16_t(wanted_indices_down)));
			//const __m256i tmp_down   = _mm256_i64gather_epi64(((uint64_t *) ptr), lower_down, 8);

			// store up
			const __m256i tmp_up1    = _mm256_lddqu_si256((__m256i *) (L + ctr));
			// TODO probably this needs to be done with a loop up table.[
			//const __m256i tmp_up     = _mm256_permute4x64_epi64(tmp_up1, indices_up);

			// gt mask still wrong
			//_mm256_maskstore_epi64((long long *)ptr, gt_mask, tmp_up);
		}

		if (wt > 0b0001111) {
			const uint64_t wanted_indices_down2 = _pext_u64(identity_indices, expanded_mask >> 8u);
			const uint64_t wanted_indices_up2 = _pdep_u64(identity_indices, expanded_mask >> 8u);
		}

		return ctr + __builtin_popcount(wt);
	}

	/// copies `from` to `to` if the bit in `gt_mask` is set.
	/// \param gt_mask
	/// \param to
	/// \param from
	/// \return
	template<const uint32_t limit>
	size_t swap(int wt, Element *to, Element *from) {
		ASSERT(wt < (1u<<limit));

		uint32_t nctr = 0;

		#pragma unroll
		for (uint32_t i = 0; i < limit; ++i) {
			if (wt & 1u) {
				std::swap(to[nctr++], from[i]);
			}

			wt >>= 1u;
		}

		return nctr;
	}

	/// NOTE: assumes T=uint64
	/// NOTE: only matches weight dk on uint32
	/// \param: e1 end index
	/// \param: random value z
	template<const uint32_t limb>
	size_t avx2_sort_nn_on32(const size_t e1,
					    	 const uint32_t z,
							 Element *L) {
		//ASSERT(n <= 64);
		//ASSERT(n > 32);
		ASSERT(limb <= ELEMENT_NR_LIMBS);

		/// just a shorter name, im lazy.
		constexpr uint32_t enl = ELEMENT_NR_LIMBS;

		const size_t s1 = 0;
		const __m256i z256 = _mm256_set1_epi32(z);
		const __m256i mask = _mm256_set1_epi32(dk+1);
		const __m256i offset = _mm256_setr_epi32(0*enl, 1*enl, 2*enl, 3*enl,
		                                         4*enl, 5*enl, 6*enl, 7*enl);
		size_t ctr = 0;

		/// NOTE: i need 2 ptr tracking the current position, because of the
		/// limb shift
		Element *ptr = (Element *)(((uint8_t *)L) + limb*4);
		Element *org_ptr = L;
	
		for (size_t i = s1; i < (e1+7)/8; i++, ptr += 8, org_ptr += 8) {
			const __m256i ptr_tmp = _mm256_i32gather_epi32(ptr, offset, 8);
			const __m256i tmp = _mm256_xor_si256(ptr_tmp, z256);
			const __m256i tmp_pop = popcount_avx2_32(tmp);
	
			const __m256i gt_mask = _mm256_cmpgt_epi32(mask, tmp_pop);
			const int wt = _mm256_movemask_ps((__m256) gt_mask);

			// now `wt` contains the incises of matches. Meaning if bit 1 in `wt` is set (and bit 0 not),
			// we need to swap the second (0 indexed) uint64_t from L + ctr with the first element from L + i.
			// The core problem is, that we need 64bit indices and not just 32bit
			if (wt) {
				ctr += swap<8>(wt, L + ctr, org_ptr);
			}
		}
	
		return ctr;
	}

	/// NOTE: assumes T=uint64
	/// NOTE: only matches weight dk on uint64_t
	/// \param: e1 end index
	/// \param: random value z
	template<const uint32_t limb>
	size_t avx2_sort_nn_on64(const size_t e1,
					    	 const uint64_t z,
							 Element *L) {
		//ASSERT(n <= 64);
		//ASSERT(n > 32);
		ASSERT(limb <= ELEMENT_NR_LIMBS);

		/// just a shorter name, im lazy.
		constexpr uint32_t enl = ELEMENT_NR_LIMBS;

		const size_t s1 = 0;
		const __m256i z256 = _mm256_set1_epi64x(z);
		const __m256i mask = _mm256_set1_epi64x(dk+1);
		const __m256i offset = _mm256_setr_epi64x(0*enl, 1*enl, 2*enl, 3*enl);
		                                         
		size_t ctr = 0;

		/// NOTE: i need 2 ptr tracking the current position, because of the
		/// limb shift
		Element *ptr = (Element *)(((uint8_t *)L) + limb*8);
		Element *org_ptr = L;

		#pragma unroll 4
		for (size_t i = s1; i < (e1+3)/4; i++, ptr += 4, org_ptr += 4) {
			const __m256i ptr_tmp = _mm256_i64gather_epi64(ptr, offset, 8);
			const __m256i tmp = _mm256_xor_si256(ptr_tmp, z256);
			const __m256i tmp_pop = popcount_avx2_64(tmp);
	
			const __m256i gt_mask = _mm256_cmpgt_epi64(mask, tmp_pop);
			const int wt = _mm256_movemask_pd((__m256) gt_mask);

			// now `wt` contains the incises of matches. Meaning if bit 1 in `wt` is set (and bit 0 not),
			// we need to swap the second (0 indexed) uint64_t from L + ctr with the first element from L + i.
			// The core problem is, that we need 64bit indices and not just 32bit
			if (wt) {
				ctr += swap<4>(wt, L + ctr, org_ptr);
			}
		}
	
	
		return ctr;
	}

	///
	/// \tparam level
	/// \param e1
	/// \param e2
	template<const uint32_t level>
	void avx2_nn_internal(const size_t e1, const size_t e2) {
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(e2 <= LIST_SIZE);

		/// NOTE: is this really the only wat to get around the restriction
		/// of partly specialized template functions?
		if constexpr (level == 0) {
			if constexpr (32 < n and n <= 64) {
				bruteforce_avx2_64_uxv<4, 4>(e1, e2);
			} else if constexpr (64 < n and n <= 128){
				bruteforce_avx2_128(e1, e2);
			} else if constexpr (128 < n and n <= 256) {
				//bruteforce_avx2_256(e1, e2);
				bruteforce_avx2_256_32_ux8<8>(e1, e2);
			} else {
				ASSERT(false);
			}
		} else {
			size_t new_e1;
			size_t new_e2;	

			if constexpr (k == 64) {
				const uint64_t z = fastrandombytes_uint64();
				new_e1 = avx2_sort_nn_on64<r - level>(e1, z, L1);
				new_e2 = avx2_sort_nn_on64<r - level>(e2, z, L2);
			} else if constexpr (k == 32) {
				const uint32_t z = (uint32_t) fastrandombytes_uint64();
				new_e1 = avx2_sort_nn_on32<r - level>(e1, z, L1);
				new_e2 = avx2_sort_nn_on32<r - level>(e2, z, L2);
			}

			ASSERT(new_e1 <= LIST_SIZE);
			ASSERT(new_e2 <= LIST_SIZE);
			ASSERT(new_e1 <= e1);
			ASSERT(new_e2 <= e2);

			if (new_e1 == 0 or new_e2 == 0) {
				return;
			}

			if ((new_e1 < BRUTEFORCE_THRESHHOLD) || (new_e2 < BRUTEFORCE_THRESHHOLD)) {
				if constexpr (32 < n and n <= 64) {
					bruteforce_avx2_64_uxv<4, 4>(e1, e2);
				} else if constexpr (64 < n and n <= 128){
					bruteforce_avx2_128(e1, e2);
				} else if constexpr (128 < n and n <= 256) {
					//bruteforce_avx2_256(e1, e2);
					bruteforce_avx2_256_32_ux8<8>(e1, e2);
				} else {
					ASSERT(false);
				}

				return;
			}

			for (uint32_t i = 0; i < N; i++) {
				avx2_nn_internal<level - 1>(new_e1, new_e2);

				if (solutions_nr) {
					std::cout << level << " " << i << " " << new_e1 << " " << new_e2 << "\n";
					break;
				}
			}
		}
	}

	template<const uint32_t level>
	void avx2_nn(const size_t e1, const size_t e2) {
		for (size_t i = 0; i < N; ++i) {
			if constexpr(32 < n and n <= 256) {
				avx2_nn_internal<level>(e1, e2);
	 		} else {
				ASSERT(false);
			}
			if (solutions_nr > 0) {
				std::cout << "outer: " << i << "\n";
				break;
			}
		}
	}

	/// simple test function
	bool run() {
		return 1;
		//Element *ptr = (Element *) malloc(10 * sizeof(T) * ELEMENT_NR_LIMBS);
		//for (int i = 0; i < 2; ++i) {
		//	ptr[i][0] = i;
		//}
		//const __m256i tmpa = shuffle_down_64(0b1010);
		//const __m256i tmpb = shuffle_up_64(0b1010);
		//__m256i tmp1, tmp2;
		//shuffle_down_2_64(tmp1, tmp2, 0b10100101);
		//shuffle_up_2_64(tmp1, tmp2, 0b10100101);

		//const __m256i gt_mask = _mm256_setr_epi32(-1, 0, -1, 0, -1, 0, 0, -1);
		//int wt = _mm256_movemask_ps((__m256) gt_mask);
		//swap_64(wt, ptr, ptr);
		//free(ptr);

		random_seed(time(NULL));
		generate_random_instance();
		
		//bruteforce_32(LIST_SIZE, LIST_SIZE);
		//bruteforce_avx2_32(LIST_SIZE, LIST_SIZE);
		//bruteforce_avx2_64(LIST_SIZE, LIST_SIZE);
		//bruteforce_avx2_64_1x1(LIST_SIZE, LIST_SIZE);
		//bruteforce_avx2_64_uxv<4,4>(LIST_SIZE, LIST_SIZE);
		//bruteforce_avx2_64_uxv_shuffle<4,4>(LIST_SIZE, LIST_SIZE);
		//bruteforce_avx2_128(LIST_SIZE, LIST_SIZE);
		//bruteforce_avx2_256(LIST_SIZE, LIST_SIZE);
		//bruteforce_avx2_256_ux4<8>(LIST_SIZE, LIST_SIZE);
		//bruteforce_avx2_256_32_ux8<2>(LIST_SIZE, LIST_SIZE);
		bruteforce_avx2_256_32_8x8(LIST_SIZE, LIST_SIZE);

		//avx2_nn64_on32<r>(LIST_SIZE, LIST_SIZE);
		//avx2_nn<r>(LIST_SIZE, LIST_SIZE);
		bool correct = all_solutions_correct();
		if (solutions_nr == 0) {
			std::cout << "no sol\n";
		}

		if (correct == 0) {
			std::cout << "wrong\n";
		}

		ASSERT(correct);
		ASSERT(solutions_nr == 1);
		return correct;
	}

	uint64_t bench() {
		random_seed(time(NULL));
		generate_random_instance();
		uint64_t ret = 0;

		clock_t t = clock();
		//bruteforce_avx2_64(LIST_SIZE, LIST_SIZE);
		//std::cout << "bruteforce_avx2_64: " << double(clock() - t)/CLOCKS_PER_SEC << "\n";
		//ret += solutions_nr;
		//solutions_nr = 0;

		//t = clock();
		//bruteforce_avx2_128(LIST_SIZE, LIST_SIZE);
		//std::cout << "bruteforce_avx2_128: " << double(clock() - t)/CLOCKS_PER_SEC << "\n";
		//ret += solutions_nr;
		//solutions_nr = 0;

		//t = clock();
		//bruteforce_avx2_256(LIST_SIZE, LIST_SIZE);
		//std::cout << "bruteforce_avx2_256: " << double(clock() - t)/CLOCKS_PER_SEC << "\n";
		//ret += solutions_nr;
		//solutions_nr = 0;

		//t = clock();
		//bruteforce_avx2_256_ux4<2>(LIST_SIZE, LIST_SIZE);
		//std::cout << "bruteforce_avx2_256_ux4<2>: " << double(clock() - t)/CLOCKS_PER_SEC << "\n";
		//ret += solutions_nr;
		//solutions_nr = 0;

		//t = clock();
		//bruteforce_avx2_256_ux4<4>(LIST_SIZE, LIST_SIZE);
		//std::cout << "bruteforce_avx2_256_ux4<4>: " << double(clock() - t)/CLOCKS_PER_SEC << "\n";
		//ret += solutions_nr;
		//solutions_nr = 0;

		//t = clock();
		//bruteforce_avx2_256_ux4<8>(LIST_SIZE, LIST_SIZE);
		//std::cout << "bruteforce_avx2_256_ux4<8>: " << double(clock() - t)/CLOCKS_PER_SEC << "\n";
		//ret += solutions_nr;
		//solutions_nr = 0;

		//t = clock();
		//bruteforce_avx2_256_32_ux8<2>(LIST_SIZE, LIST_SIZE);
		//std::cout << "bruteforce_avx2_256_32_ux8<2>: " << double(clock() - t)/CLOCKS_PER_SEC << "\n";
		//ret += solutions_nr;
		//solutions_nr = 0;

		//t = clock();
		//bruteforce_avx2_256_32_ux8<4>(LIST_SIZE, LIST_SIZE);
		//std::cout << "bruteforce_avx2_256_32_ux8<4>: " << double(clock() - t)/CLOCKS_PER_SEC << "\n";
		//ret += solutions_nr;
		//solutions_nr = 0;

		t = clock();
		bruteforce_avx2_256_32_ux8<8>(LIST_SIZE, LIST_SIZE);
		std::cout << "bruteforce_avx2_256_32_ux8<8>: " << double(clock() - t)/CLOCKS_PER_SEC << "\n";
		ret += solutions_nr;
		solutions_nr = 0;

		t = clock();
		bruteforce_avx2_256_32_8x8(LIST_SIZE, LIST_SIZE);
		std::cout << "bruteforce_avx2_256_32_8x8: " << double(clock() - t)/CLOCKS_PER_SEC << "\n";
		ret += solutions_nr;
		solutions_nr = 0;

		//t = clock();
		//bruteforce_avx2_256_v2(LIST_SIZE, LIST_SIZE);
		//std::cout << "bruteforce_avx2_256_v2: " << double(clock() - t)/CLOCKS_PER_SEC << "\n";
		//ret += solutions_nr;
		//solutions_nr = 0;

		//t = clock();
		//bruteforce_avx2_64_1x1(LIST_SIZE, LIST_SIZE);
		//std::cout << "avx_64_1x1: " << double(clock() - t)/CLOCKS_PER_SEC << "\n";
		//ret += solutions_nr;
		//solutions_nr = 0;

		//t = clock();
		//bruteforce_avx2_64_uxv<4,4>(LIST_SIZE, LIST_SIZE);
		//std::cout << "avx_64_4x4: " << double(clock() - t)/CLOCKS_PER_SEC << "\n";
		//ret += solutions_nr;
		//solutions_nr = 0;
		
		//t = clock();
		//bruteforce_avx2_64_uxv<8,8>(LIST_SIZE, LIST_SIZE);
		//std::cout << "avx_64_8x8: " << double(clock() - t)/CLOCKS_PER_SEC << "\n";
		//ret += solutions_nr;
		//solutions_nr = 0;

		//t = clock();
		//bruteforce_avx2_64_uxv_shuffle<4,4>(LIST_SIZE, LIST_SIZE);
		//std::cout << "avx_64_4x4_shuffle: " << double(clock() - t)/CLOCKS_PER_SEC << "\n";
		//ret += solutions_nr;
		//solutions_nr = 0;

		//t = clock();
		//bruteforce_avx2_64_uxv_shuffle<8,8>(LIST_SIZE, LIST_SIZE);
		//std::cout << "avx_64_8x8_shuffle: " << double(clock() - t)/CLOCKS_PER_SEC << "\n";
		//ret += solutions_nr;
		//solutions_nr = 0;

		t = clock();
		avx2_nn<r>(LIST_SIZE, LIST_SIZE);
		std::cout << "avx2_nn<2>: " << double(clock() - t)/CLOCKS_PER_SEC << "\n";
		ret += solutions_nr;
		std::cout << "sols: " << solutions_nr << "\n";
		return ret;
	}
};

#endif//NN_CODE_WINDOWED_AVX2_H
