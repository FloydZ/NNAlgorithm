#include <stdint.h>
#include <immintrin.h>

constexpr uint32_t NR_T_LIMBS = 2;
constexpr uint32_t n = 96;
constexpr uint32_t delta = 12;
using T = uint64_t;
using Element = T[4];

Element *L1, *L2;

void find_solution(const size_t i,
		           const size_t j) {
}

__m256i popcount(__m256i tmp) {
	return tmp;
}

/// returns a permutatio that shuffles down a mask
__m256i shuffle_down(const __m256i mask) {
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

void bruteforce_128(const size_t e1,
					const size_t e2) {
	const size_t s1 = 0;
	const size_t s2 = 0;

	for (size_t i = s1; i < e1; i++) {
		for (size_t j = s2; j < e2; j++) {
			uint32_t weight = 0;

			#pragma unroll
			for (uint32_t k = 0; k < NR_T_LIMBS; k++){
				weight += __builtin_popcountll(L1[i][k] ^ L2[j][k]);
			}

			if (weight == 0) {
				find_solution(i, j);
			}
		}
	}
}


/// NOTE: assumes T=uint64
/// :param e1 end index
/// :param random value z
size_t avx2_sort_nn64_on32(const size_t e1,
				    	 const uint32_t z,
						 Element *L) {
	constexpr uint32_t limb = 0;
	const size_t s1 = 0;
	const __m256i z256 = _mm256_set1_epi32(z);
	const __m256i mask = _mm256_set1_epi32(delta+1);
	const __m256i offset = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	size_t ctr = 0;


	Element *ptr = L;

	for (size_t i = s1; i < e1; i++, ptr++) {
		const __m256i ptr_tmp = _mm256_i32gather_epi32(ptr, offset, 8);
		const __m256i tmp = _mm256_xor_si256(ptr_tmp, z256);
		const __m256i tmp_pop = popcount(tmp);

		const __m256i gt_mask = _mm256_cmpgt_epi32(mask, tmp_pop);
		const int wt = _mm256_movemask_ps((__m256) gt_mask);

		if (wt) {
			const __m256i shuffle = shuffle_down(ptr_tmp);
			const __m256i shf_mask = _mm256_permutevar8x32_ps(gt_mask, shuffle);
			const __m256i shf_ptr_tmp = _mm256_permutevar8x32_ps(ptr_tmp, shuffle);
			
			// TODO think about if this is current because it write back 64 bits: _mm256_maskstore_epi32
			__m256i xor_tmp = _mm256_lddqu_si256((__m256i *)(L + ctr));
			_mm256_maskstore_epi64((__int64_t *)ptr, shf_mask, xor_tmp);
			_mm256_maskstore_epi64((__int64_t *)(L + ctr), shf_mask, shf_ptr_tmp);

			ctr += __builtin_popcount(wt);
		}
	}


	return ctr;
}
