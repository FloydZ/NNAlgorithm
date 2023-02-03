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

// good values
//      n   N   r   dk   L      B
//      64  25  2   11  20      432

class WindowedAVX2 {
public:
	// TODO well make them instantiatable via constexpr constructor
	// Additional parameters which are not defined in the base class.

	constexpr static size_t n = 64;         // n
	constexpr static size_t r = 2;           // number of blocks
	constexpr static size_t N = 50;          // number of list spawn
	//constexpr static size_t LIST_SIZE = 112;// well, list size
	constexpr static size_t LIST_SIZE = 16*1u<<10u;// well, list size
	constexpr static double d_ = 0.1;       // delta/n
	constexpr static uint64_t k = 32;       // n/r BlockSize
	constexpr static uint64_t dk = 11;      // weight per block
	constexpr static uint64_t epsilon = 0;  // additional offset/variance we allow in each level to mach on.
	
	constexpr static uint64_t BRUTEFORCE_THRESHHOLD = 108*4;

	// Array indicating the window boundaries.
	std::vector<uint64_t> buckets_windows{0, 32};

	// How many solution did we already searched in the golden NN search.
	uint64_t solution_searched = 0;

	/// Base types
	using T = uint64_t; // NOTE do not change.
	constexpr static size_t T_BITSIZE = sizeof(T) * 8;
	constexpr static size_t ELEMENT_NR_LIMBS = (n + T_BITSIZE - 1) / T_BITSIZE;
	using Element = T[ELEMENT_NR_LIMBS];
	using List    = std::array<Element, LIST_SIZE>;

	// instance
	// alignas(64) List L1, L2;
	alignas(64) Element *L1, *L2;

	// solution
	size_t solution_l = 0, solution_r = 0, solutions_nr = 0;
	std::vector<std::pair<size_t, size_t>> solutions;



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
		solution_l = fastrandombytes_uint64() % LIST_SIZE;
		solution_r = fastrandombytes_uint64() % LIST_SIZE;

		Element sol;
		generate_random_element(sol);

		// inject the solution
		for (uint32_t i = 0; i < ELEMENT_NR_LIMBS; ++i) {
			L1[solution_l][i] = sol[i];
			L2[solution_r][i] = sol[i];
		}
	}
	
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

	/// special popcount, which popcounts on 8 * 32u bit limbs in parallel
	static __m256i popcount_avx2_32(const __m256i vec) noexcept {
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
	
		// not the best
		const __m256i mask = _mm256_set1_epi32(0xff);
		__m256i ret = _mm256_and_si256(local, mask);
		ret = _mm256_add_epi8(ret, _mm256_and_si256(_mm256_srli_epi32(local,  8), mask));
		ret = _mm256_add_epi8(ret, _mm256_and_si256(_mm256_srli_epi32(local, 16), mask));
		ret = _mm256_add_epi8(ret, _mm256_and_si256(_mm256_srli_epi32(local, 24), mask));
		return ret;
	}
	
	/// special popcount which popcounts on 4 * 64 bit limbs in parallel
	static __m256i popcount_avx2_64(const __m256i vec) noexcept {
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
		solutions.resize(solutions_nr + 1);
		solutions[solutions_nr++] = std::pair<size_t, size_t>{li, lr};
	}

	// checks whether all submitted solutions are correct
	bool all_solutions_correct() {
		if (solutions_nr == 0)
			return false;
		
		for (uint32_t i = 0; i < solutions_nr; i++) {
			bool equal = true;
			for (uint32_t j = 0; j < ELEMENT_NR_LIMBS; j++) {
				//std::cout << L1[solutions[i].first][j] << " " << L2[solutions[i].second][j] << "\n";
				equal &= L1[solutions[i].first][j] == L2[solutions[i].second][j];
			}

			if (!equal)
				return false;
		}

		return true;
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: only compares a single 32 bit column of the list. But its
	///			still possible
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 23...43) is impossible.
	/// NOTE: uses avx2
	/// NOTE: only a single 32bit element is compared.
	/// \param s1 start index of list 1
	/// \param e1 end index of list 1
	/// \param s2 start index list 2
	/// \param e2 end index list 2
	void bruteforce_avx2_32(const size_t e1,
	                        const size_t e2) noexcept {
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// limb position to compare on, basically the column to compare on.
		constexpr uint32_t limb_pos = 0;

		/// strive is the number of bytes to the next element
		constexpr uint32_t stride = 8;

		/// difference of the memory location in the right list
		const __m256i loadr = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

		/// allowed weight to match on
		const __m256i weight = _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0);

		/// NOTE: only possible because L2 is a continuous memory block
		const T *ptr_r = (T *)L2;

		for (size_t i = s1; i < e1; ++i) {
			// NOTE: implicit typecast because T = uint64
			const __m256i li = _mm256_set1_epi32(L1[i][limb_pos]);

			for (size_t j = s2; j < s2+(e2+7)/8; ++j) {
				const __m256i ri = _mm256_i32gather_epi32(ptr_r + 8*j, loadr, stride);
				const __m256i tmp1 = _mm256_xor_si256(li, ri);
				const __m256i tmp2 = _mm256_cmpeq_epi32(tmp1, weight);
				const int m = _mm256_movemask_ps((__m256) tmp2);

				if (m) {
					found_solution(i, j+__builtin_popcount(m));
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
				const __m256i tmp1 = _mm256_xor_si256(li, ri);
				const __m256i tmp2 = _mm256_cmpeq_epi64(tmp1, weight);
				const int m = _mm256_movemask_pd((__m256d) tmp2);

				if (m) {
					found_solution(i, j+__builtin_ctz(m));
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
				const __m256i tmp1 = _mm256_xor_si256(li1, ri);
				const __m256i tmp2 = _mm256_cmpeq_epi64(tmp1, weight);
				const int m1 = _mm256_movemask_pd((__m256d) tmp2);

				if (m1) {
					const __m256i ri = _mm256_i32gather_epi64(ptr_r, loadr2, 8);
					const __m256i tmp1 = _mm256_xor_si256(li2, ri);
					const __m256i tmp2 = _mm256_cmpeq_epi64(tmp1, weight);
					const int m1 = _mm256_movemask_pd((__m256d) tmp2);

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
				const __m256i tmp1 = _mm256_xor_si256(li1, ri);
				const __m256i tmp2 = _mm256_cmpeq_epi64(tmp1, weight);
				const int m1 = _mm256_movemask_pd((__m256d) tmp2);

				if (m1) {
					const __m256i ri = _mm256_i32gather_epi64(ptr_r, loadr2, 8);
					const __m256i tmp1 = _mm256_xor_si256(li2, ri);
					const __m256i tmp2 = _mm256_cmpeq_epi64(tmp1, weight);
					const int m1 = _mm256_movemask_pd((__m256d) tmp2);

					if (m1) {
						const __m256i ri = _mm256_i32gather_epi64(ptr_r, loadr3, 8);
						const __m256i tmp1 = _mm256_xor_si256(li3, ri);
						const __m256i tmp2 = _mm256_cmpeq_epi64(tmp1, weight);
						const int m1 = _mm256_movemask_pd((__m256d) tmp2);

						if (m1) {
							const __m256i ri = _mm256_i32gather_epi64(ptr_r, loadr4, 8);
							const __m256i tmp1 = _mm256_xor_si256(li4, ri);
							const __m256i tmp2 = _mm256_cmpeq_epi64(tmp1, weight);
							const int m1 = _mm256_movemask_pd((__m256d) tmp2);
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
				const __m256i tmp1 = _mm256_xor_si256(li, ri);
				const __m256i tmp2 = _mm256_cmpeq_epi64(tmp1, weight);
				const int m = _mm256_movemask_pd((__m256d) tmp2);

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
						const __m256i t1 = _mm256_xor_si256(tmp1, tmp2);
						const __m256i t2 = _mm256_cmpeq_epi64(t1, weight);
						const int m = _mm256_movemask_pd((__m256d) t2);

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

						__m256i t1 = _mm256_xor_si256(tmp1, tmp2);
						__m256i t2 = _mm256_cmpeq_epi64(t1, weight);
						int m = _mm256_movemask_pd((__m256d) t2);
						if (m) {
							const size_t jprime = j*4 + a2*4 + __builtin_ctz(m);
							const size_t iprime = i*4 + a1*4 + __builtin_ctz(m);
							//std::cout << L1[iprime][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
							found_solution(iprime, jprime);
						}

						tmp2 = _mm256_permute4x64_epi64(tmp2, 0b10010011);
						t1 = _mm256_xor_si256(tmp1, tmp2);
						t2 = _mm256_cmpeq_epi64(t1, weight);
						m = _mm256_movemask_pd((__m256d) t2);
						if (m) {
							const size_t jprime = j*4 + a2*4 + __builtin_ctz(m) + 3;
							const size_t iprime = i*4 + a1*4 + __builtin_ctz(m);
							//std::cout << L1[iprime][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
							found_solution(iprime, jprime);
						}

						tmp2 = _mm256_permute4x64_epi64(tmp2, 0b10010011);
						t1 = _mm256_xor_si256(tmp1, tmp2);
						t2 = _mm256_cmpeq_epi64(t1, weight);
						m = _mm256_movemask_pd((__m256d) t2);
						if (m) {
							const size_t jprime = j*4 + a2*4 + __builtin_ctz(m) + 2;
							const size_t iprime = i*4 + a1*4+ __builtin_ctz(m);
							//std::cout << L1[iprime][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
							found_solution(iprime, jprime);
						}

						tmp2 = _mm256_permute4x64_epi64(tmp2, 0b10010011);
						t1 = _mm256_xor_si256(tmp1, tmp2);
						t2 = _mm256_cmpeq_epi64(t1, weight);
						m = _mm256_movemask_pd((__m256d) t2);
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
	size_t swap(int wt, Element *to, Element *from) {
		ASSERT(wt < (1u<<8u));

		uint32_t nctr = 0;

		#pragma unroll
		for (uint32_t i = 0; i < 8; ++i) {
			if (wt & 1u) {
				std::swap(to[nctr++], from[i]);
			}

			wt >>= 1u;
		}

		return nctr;
	}

	/// NOTE: assumes T=uint64
	/// NOTE:
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
				ctr += swap(wt, L + ctr, org_ptr);
			}
		}
	
	
		return ctr;
	}

	///
	/// \tparam level
	/// \param e1
	/// \param e2
	template<const uint32_t level>
	void avx2_nn_on32_internal(const size_t e1, const size_t e2) {
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(e2 <= LIST_SIZE);

		/// NOTE: is this really the only wat to get around the restriction
		/// of partly specialized template functions?
		if constexpr (level == 0) {
			if constexpr (32 < n and n <= 64) {
				bruteforce_avx2_64_uxv<4, 4>(e1, e2);
			} else if constexpr (64 < n and n <= 128){
				bruteforce_avx2_128(e1, e2);
			}
		} else {
			const uint32_t z = (uint32_t) fastrandombytes_uint64();
			const size_t new_e2 = avx2_sort_nn_on32<r - level>(e2, z, L2);
			const size_t new_e1 = avx2_sort_nn_on32<r - level>(e1, z, L1);
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
				}
				return;
			}

			for (uint32_t i = 0; i < N; i++) {
				avx2_nn_on32_internal<level - 1>(new_e1, new_e2);

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
			if constexpr(32 < n and n <= 128) {
				avx2_nn_on32_internal<level>(e1, e2);
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
		Element *ptr = (Element *) malloc(10 * sizeof(T) * ELEMENT_NR_LIMBS);
		for (int i = 0; i < 2; ++i) {
			ptr[i][0] = i;
		}
		//const __m256i tmpa = shuffle_down_64(0b1010);
		//const __m256i tmpb = shuffle_up_64(0b1010);
		//__m256i tmp1, tmp2;
		//shuffle_down_2_64(tmp1, tmp2, 0b10100101);
		//shuffle_up_2_64(tmp1, tmp2, 0b10100101);

		//const __m256i gt_mask = _mm256_setr_epi32(-1, 0, -1, 0, -1, 0, 0, -1);
		//int wt = _mm256_movemask_ps((__m256) gt_mask);
		//swap_64(wt, ptr, ptr);
		free(ptr);

		random_seed(time(NULL));
		generate_random_instance();

		//bruteforce_avx2_32(LIST_SIZE, LIST_SIZE);
		//bruteforce_avx2_64(LIST_SIZE, LIST_SIZE);
		//bruteforce_avx2_128(LIST_SIZE, LIST_SIZE);
		//bruteforce_avx2_256(LIST_SIZE, LIST_SIZE);
		//bruteforce_avx2_64_1x1(LIST_SIZE, LIST_SIZE);
		//bruteforce_avx2_64_uxv<4,4>(LIST_SIZE, LIST_SIZE);
		//bruteforce_avx2_64_uxv_shuffle<4,4>(LIST_SIZE, LIST_SIZE);
		//avx2_nn64_on32<r>(LIST_SIZE, LIST_SIZE);
		avx2_nn<r>(LIST_SIZE, LIST_SIZE);
		bool correct = all_solutions_correct();
		if (solutions_nr == 0 or !correct) {
			std::cout << "wrong\n";
		}

		ASSERT(solutions_nr == 1);
		ASSERT(correct);
		return correct;
	}

	uint64_t bench() {
		return 1;
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

		//t = clock();
		//avx2_nn64_on32<r>(LIST_SIZE, LIST_SIZE);
		//std::cout << "avx2_nn64_on32<2>: " << double(clock() - t)/CLOCKS_PER_SEC << "\n";
		//ret += solutions_nr;
		//std::cout << "sols: " << solutions_nr << "\n";
		return ret;
	}
};

#endif//NN_CODE_WINDOWED_AVX2_H
