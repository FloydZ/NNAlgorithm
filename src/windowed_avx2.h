#ifndef NN_CODE_WINDOWED_AVX2_H
#define NN_CODE_WINDOWED_AVX2_H

#include <cstdint>
#include <cstdlib>
#include "immintrin.h"

#include "random.h"

#define LOAD_ALIGNED
#ifdef LOAD_ALIGNED
#define LOAD256(x) _mm256_lddqu_si256(x)
#else
#define LOAD256(x) _mm256_load_si256(x)
#endif

class WindowedAVX2 {
public:
	// TODO well make them instanciateable via costexpr constructor
	// Additional parameters which are not defined in the base class.

	constexpr static size_t n = 64;         // n
	constexpr static size_t r = 1;          // number of blocks
	constexpr static size_t N = 1;          // number of list spwan
	//constexpr static size_t LIST_SIZE = 104;// well, list size
	constexpr static size_t LIST_SIZE = 8*1u<<14u;// well, list size
	constexpr static double d_ = 0.1;       // delta/n
	constexpr static uint64_t k = 32;       // n/r BlockSize
	constexpr static uint64_t dk = 12;      // weight per block
	constexpr static uint64_t epsilon = 0;  // additional offset/variance we allow in each level to mach on.

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

	///
	/// \param L
	static void generate_random_lists(Element *L) noexcept {
		for (size_t i = 0; i < LIST_SIZE; i++) {
			generate_random_element(L[i]);
		}
	}

	///
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


	void found_solution(const size_t li,
	                    const size_t lr) {
		solutions.resize(solutions_nr + 1);
		solutions[solutions_nr++] = std::pair<size_t, size_t>{li, lr};
	}

	/// bruteforce the two lists between the given start and end indicis.
	/// NOTE: only compares a single 32 bit column of the list. But its
	///			still possible
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 23...43) is impossible.
	/// NOTE: uses avx2
	/// NOTE: only a single 32bit element is compared.
	/// \param s1 start index of list 1
	/// \param e1 end index of list 1
	/// \param s2 start index list 2
	/// \param e2 end index list 2
	void bruteforce_avx2_32(const size_t s1,
	                        const size_t e1,
	                        const size_t s2,
	                        const size_t e2) noexcept {
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


	/// bruteforce the two lists between the given start and end indicis.
	/// NOTE: without avx2
	/// \param s1 start index of list 1
	/// \param e1 end index of list 1
	/// \param s2 start index list 2
	/// \param e2 end index list 2
	void bruteforce_64(const size_t s1,
	                   const size_t e1,
	                   const size_t s2,
	                   const size_t e2) noexcept {
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// limb position to compare on, basically the column to compare on.
		constexpr uint32_t limb_pos = 0;

		for (size_t i = s1; i < e1; ++i) {
			for (size_t j = s2; j < s2+e2; ++j) {
				if (L1[i] == L2[j]) {
					found_solution(i, j);
				}
			}
		}
	}

	/// bruteforce the two lists between the given start and end indicis.
	/// NOTE: uses avx2
	/// NOTE: only compares a single 64 bit column of the list. But its
	///			still possible
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// \param s1 start index of list 1
	/// \param e1 end index of list 1
	/// \param s2 start index list 2
	/// \param e2 end index list 2
	void bruteforce_avx2_64(const size_t s1,
	                        const size_t e1,
	                        const size_t s2,
	                        const size_t e2) noexcept {
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

			for (size_t j = s2; j < s2+e2; ++j, ptr_r += 4) {
				const __m256i ri = _mm256_i32gather_epi64(ptr_r, loadr, stride);
				const __m256i tmp1 = _mm256_xor_si256(li, ri);
				const __m256i tmp2 = _mm256_cmpeq_epi32(tmp1, weight);
				const int m = _mm256_movemask_ps((__m256) tmp2);

				if (m) {
					found_solution(i, j+__builtin_popcount(m));
				}
			}
		}
	}

	/// NOTE: in comparison to the other version `bruteforce_avx2_64` this implementation
	///			assumes that the elements to compare are fully compared on all n variables
	///  		e.g. ELEMENT_NR_LIMBS == 1
	/// \param s1 start index of list 1
	/// \param e1 end index of list 1
	/// \param s2 start index list 2
	/// \param e2 end index list 2
	void bruteforce_avx2_64_1x1(const size_t s1,
	                            const size_t e1,
	                            const size_t s2,
	                            const size_t e2) noexcept {
		static_assert(ELEMENT_NR_LIMBS == 1);
		static_assert(n <= 64);
		static_assert(n >= 32);
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
				const __m256i tmp2 = _mm256_cmpeq_epi32(tmp1, weight);
				const int m = _mm256_movemask_ps((__m256) tmp2);

				if (m) {
					found_solution(i, j+__builtin_popcount(m));
				}
			}
		}
	}

	/// NOTE: in comparison to the other version `bruteforce_avx2_64` this implementation
	///			assumes that the elements to compare are fully compared on all n variables
	///  		e.g. ELEMENT_NR_LIMBS == 1
	/// NOTE: compared to `bruteforce_avx2_64_1x1` this unrolls `u` elementes in the left
	///			list and `v` elements on the right.
	/// \param s1 start index of list 1
	/// \param e1 end index of list 1
	/// \param s2 start index list 2
	/// \param e2 end index list 2
	template<const uint32_t u, const uint32_t v>
	void bruteforce_avx2_64_uxv(const size_t s1,
	                            const size_t e1,
	                            const size_t s2,
	                            const size_t e2) noexcept {
		static_assert(ELEMENT_NR_LIMBS == 1);
		static_assert(n <= 64);
		static_assert(n >= 32);
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
						const __m256i t2 = _mm256_cmpeq_epi32(t1, weight);
						const int m = _mm256_movemask_ps((__m256) t2);
						if (m) {
							found_solution(i, j+__builtin_popcount(m));
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
	/// \param s1 start index of list 1
	/// \param e1 end index of list 1
	/// \param s2 start index list 2
	/// \param e2 end index list 2
	template<const uint32_t u, const uint32_t v>
	void bruteforce_avx2_64_uxv_shuffle(const size_t s1,
	                                    const size_t e1,
	                                    const size_t s2,
	                                    const size_t e2) noexcept {
		static_assert(ELEMENT_NR_LIMBS == 1);
		static_assert(n <= 64);
		static_assert(n >= 32);
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		__m256i lii[u], rii[v];
		__m256i *ptr_l = (__m256i *)L2;

		/// allowed weight to match on
		const __m256i weight = _mm256_setr_epi64x(0, 0, 0, 0);

		for (size_t i = s1; i < (e1+3)/4; i += u, ptr_l += u) {

			#pragma unroll
			for (uint32_t j = 0; j < u; ++j) {
				lii[j] = LOAD256(ptr_l + u);
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
						__m256i t2 = _mm256_cmpeq_epi32(t1, weight);
						int m = _mm256_movemask_ps((__m256) t2);
						if (m) {
							found_solution(i, j+__builtin_popcount(m));
						}

						tmp2 = _mm256_permute4x64_epi64(tmp2, 0b10010011);
						t1 = _mm256_xor_si256(tmp1, tmp2);
						t2 = _mm256_cmpeq_epi32(t1, weight);
						m = _mm256_movemask_ps((__m256) t2);
						if (m) {
							found_solution(i, j+__builtin_popcount(m));
						}

						tmp2 = _mm256_permute4x64_epi64(tmp2, 0b10010011);
						t1 = _mm256_xor_si256(tmp1, tmp2);
						t2 = _mm256_cmpeq_epi32(t1, weight);
						m = _mm256_movemask_ps((__m256) t2);
						if (m) {
							found_solution(i, j+__builtin_popcount(m));
						}

						tmp2 = _mm256_permute4x64_epi64(tmp2, 0b10010011);
						t1 = _mm256_xor_si256(tmp1, tmp2);
						t2 = _mm256_cmpeq_epi32(t1, weight);
						m = _mm256_movemask_ps((__m256) t2);
						if (m) {
							found_solution(i, j+__builtin_popcount(m));
						}
					}
				}
			}
		}
	}

	void run() {
		generate_random_instance();

		// bruteforce_avx2_32(0, LIST_SIZE, 0, LIST_SIZE);
		// bruteforce_avx2_64(0, LIST_SIZE, 0, LIST_SIZE);
		//bruteforce_avx2_64_1x1(0, LIST_SIZE, 0, LIST_SIZE);
		//bruteforce_avx2_64_uxv<4,4>(0, LIST_SIZE, 0, LIST_SIZE);
		bruteforce_avx2_64_uxv_shuffle<4,4>(0, LIST_SIZE, 0, LIST_SIZE);
		if (solutions_nr == 0) {
			std::cout << "wrong\n";
		}
		ASSERT(solutions_nr >= 1);
	}

	uint64_t bench() {
		generate_random_instance();
		uint64_t ret = 0;

		clock_t t = clock();
		bruteforce_64(0, LIST_SIZE, 0, LIST_SIZE);
		std::cout << "simple: " << double(clock() - t)/CLOCKS_PER_SEC << "\n";
		ret += solutions_nr;

		//bruteforce_avx2_64(0, LIST_SIZE, 0, LIST_SIZE);
		//std::cout << "avx_64: " << double(clock() - t)/CLOCKS_PER_SEC << "\n";
		//ret += solutions_nr;

		t = clock();
		bruteforce_avx2_64_1x1(0, LIST_SIZE, 0, LIST_SIZE);
		std::cout << "avx_64_1x1: " << double(clock() - t)/CLOCKS_PER_SEC << "\n";
		ret += solutions_nr;

		t = clock();
		bruteforce_avx2_64_uxv<4,4>(0, LIST_SIZE, 0, LIST_SIZE);
		std::cout << "avx_64_uxv: " << double(clock() - t)/CLOCKS_PER_SEC << "\n";
		ret += solutions_nr;

		t = clock();
		bruteforce_avx2_64_uxv_shuffle<4,4>(0, LIST_SIZE, 0, LIST_SIZE);
		std::cout << "avx_64_uxv_shuffle: " << double(clock() - t)/CLOCKS_PER_SEC << "\n";
		ret += solutions_nr;

		return ret;
	}
};
#endif//NN_CODE_WINDOWED_AVX2_H
