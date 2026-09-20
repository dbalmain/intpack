//! Four-lane SSE2 form of the crate's delta convention.
//!
//! [`values`] is the in-vector inclusive prefix sum of `gap + 1`, plus the
//! previous group's last lane broadcast — the same operation the bitpack
//! kernel and the Stream VByte kernel both need. SSE2 only; compiled where
//! `cfg(target_feature = "sse2")` holds, like `bitpack::simd`.

use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_set1_epi32, _mm_shuffle_epi32, _mm_slli_si128,
};

/// The four values following `prev`'s last lane, from four gaps.
#[inline]
#[target_feature(enable = "sse2")]
pub fn values(prev: __m128i, gaps: __m128i) -> __m128i {
    let x = _mm_add_epi32(gaps, _mm_set1_epi32(1));
    let x = _mm_add_epi32(x, _mm_slli_si128::<4>(x));
    let x = _mm_add_epi32(x, _mm_slli_si128::<8>(x));
    _mm_add_epi32(x, _mm_shuffle_epi32::<0xff>(prev))
}
