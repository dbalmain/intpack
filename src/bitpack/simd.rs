//! The SSE2 kernel: the four lanes of the layout are the four `u32` lanes of
//! one `__m128i`, so each `[u32; 4]` group of the block is one vector and
//! each packed word group is one vector. Words are produced and consumed in
//! stream order with the fill tracked by the group index, which is the same
//! arithmetic as `scalar::position` — `scalar` places each element from
//! scratch, this carries the partially filled word across groups. The width
//! is a const generic and the group loop is unrolled by `each_group!`, so
//! the per-group fill, and hence every shift count and word index, is a
//! constant; the shifts take their count from a register so that no
//! `generic_const_exprs` is needed, and LLVM folds them to immediates.
//!
//! SSE2 is part of the `x86_64` baseline, so there is nothing to detect. The
//! `unsafe` here is confined to the four load/store helpers, each of which
//! reinterprets a reference to exactly the 16 bytes the vector holds.

use core::arch::x86_64::{
    __m128i, _mm_and_si128, _mm_cvtsi32_si128, _mm_loadu_si128, _mm_or_si128, _mm_set1_epi32,
    _mm_setzero_si128, _mm_sll_epi32, _mm_slli_si128, _mm_srl_epi32, _mm_srli_si128,
    _mm_storeu_si128, _mm_sub_epi32,
};

use super::{BLOCK, mask};
use crate::delta;

type V = __m128i;

#[inline]
#[target_feature(enable = "sse2")]
fn load_bytes(word: &[u8; 16]) -> V {
    // SAFETY: `word` is 16 readable bytes; an unaligned load needs no more.
    unsafe { _mm_loadu_si128(word.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "sse2")]
fn load_ints(group: &[u32; 4]) -> V {
    // SAFETY: as `load_bytes`.
    unsafe { _mm_loadu_si128(group.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "sse2")]
fn store_bytes(v: V, word: &mut [u8; 16]) {
    // SAFETY: `word` is 16 writable bytes.
    unsafe { _mm_storeu_si128(word.as_mut_ptr().cast(), v) }
}

#[inline]
#[target_feature(enable = "sse2")]
fn store_ints(v: V, group: &mut [u32; 4]) {
    // SAFETY: as `store_bytes`.
    unsafe { _mm_storeu_si128(group.as_mut_ptr().cast(), v) }
}

/// Lane-wise `v << n`; unlike `u32 << n`, `n == 32` yields zero.
#[inline]
#[target_feature(enable = "sse2")]
fn shl(v: V, n: u32) -> V {
    _mm_sll_epi32(v, _mm_cvtsi32_si128(n as i32))
}

/// Lane-wise `v >> n`; `n == 32` yields zero.
#[inline]
#[target_feature(enable = "sse2")]
fn shr(v: V, n: u32) -> V {
    _mm_srl_epi32(v, _mm_cvtsi32_si128(n as i32))
}

/// Bits of the current word already filled before group `i`.
#[inline(always)]
const fn fill<const W: u32>(i: usize) -> u32 {
    (i as u32 * W) % 32
}

/// The word group `i` starts in.
#[inline(always)]
const fn word<const W: u32>(i: usize) -> usize {
    i * W as usize / 32
}

/// Runs the body once per group of the block with `i` a literal, so that
/// everything derived from it folds at compile time. LLVM leaves a 32-trip
/// loop with this much body rolled, and a rolled loop computes every shift
/// count and word index at run time.
macro_rules! each_group {
    (|$i:ident| $body:block) => {
        each_group!(@ $i $body,
            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)
    };
    (@ $i:ident $body:block, $($n:literal)*) => {
        $({
            let $i: usize = $n;
            $body
        })*
    };
}

#[target_feature(enable = "sse2")]
pub fn pack<const W: u32>(block: &[u32; BLOCK], out: &mut [u8]) {
    pack_with::<W>(block, out, |v| v);
}

#[target_feature(enable = "sse2")]
pub fn pack_sorted<const W: u32>(base: Option<u32>, block: &[u32; BLOCK], out: &mut [u8]) {
    let mut prev = _mm_set1_epi32(delta::carried(base) as i32);
    pack_with::<W>(block, out, |v| {
        let gaps = gaps(prev, v);
        prev = v;
        gaps
    });
}

#[target_feature(enable = "sse2")]
pub fn width_sorted(base: Option<u32>, block: &[u32; BLOCK]) -> u32 {
    let mut prev = _mm_set1_epi32(delta::carried(base) as i32);
    let mut all = _mm_setzero_si128();
    for group in block.as_chunks::<4>().0 {
        let v = load_ints(group);
        all = _mm_or_si128(all, gaps(prev, v));
        prev = v;
    }
    let mut lanes = [0u32; 4];
    store_ints(all, &mut lanes);
    32 - (lanes[0] | lanes[1] | lanes[2] | lanes[3]).leading_zeros()
}

/// Pack `emit(group)` for each group of the block.
#[inline]
#[target_feature(enable = "sse2")]
fn pack_with<const W: u32>(block: &[u32; BLOCK], out: &mut [u8], mut emit: impl FnMut(V) -> V) {
    let groups = block.as_chunks::<4>().0;
    let words = out.as_chunks_mut::<16>().0;
    assert_eq!(words.len(), W as usize);
    // The carry into each group: the bits of the earlier groups' values that
    // belong to the word this group starts in.
    let mut carry = _mm_setzero_si128();
    each_group!(|i| {
        let fill = fill::<W>(i);
        let v = emit(load_ints(&groups[i]));
        let acc = _mm_or_si128(carry, shl(v, fill));
        carry = if fill + W < 32 {
            acc
        } else {
            store_bytes(acc, &mut words[word::<W>(i)]);
            shr(v, 32 - fill)
        };
    });
    // The last group always ends a word, so nothing is carried out.
    let _ = carry;
}

#[target_feature(enable = "sse2")]
pub fn unpack<const W: u32>(packed: &[u8], out: &mut [u32; BLOCK]) {
    unpack_with::<W>(packed, out, |v| v);
}

#[target_feature(enable = "sse2")]
pub fn unpack_sorted<const W: u32>(base: Option<u32>, packed: &[u8], out: &mut [u32; BLOCK]) {
    let mut prev = _mm_set1_epi32(delta::carried(base) as i32);
    unpack_with::<W>(packed, out, |gaps| {
        prev = delta::simd::values(prev, gaps);
        prev
    });
}

/// Unpack each group and store `emit(group)`.
#[inline]
#[target_feature(enable = "sse2")]
fn unpack_with<const W: u32>(packed: &[u8], out: &mut [u32; BLOCK], mut emit: impl FnMut(V) -> V) {
    let words = packed.as_chunks::<16>().0;
    assert_eq!(words.len(), W as usize);
    let groups = out.as_chunks_mut::<4>().0;
    let mask = _mm_set1_epi32(mask(W) as i32);
    let mut acc = _mm_setzero_si128();
    each_group!(|i| {
        let fill = fill::<W>(i);
        if fill == 0 && W > 0 {
            acc = load_bytes(&words[word::<W>(i)]);
        }
        let mut v = shr(acc, fill);
        if fill + W > 32 {
            acc = load_bytes(&words[word::<W>(i) + 1]);
            v = _mm_or_si128(v, shl(acc, 32 - fill));
        }
        store_ints(emit(_mm_and_si128(v, mask)), &mut groups[i]);
    });
}

// ── the delta convention, four at a time ──
//
// Both directions carry `prev`, the previous group of values, in the
// `delta::carried` form: the head-of-list base is `u32::MAX`, so the first
// gap is `v - MAX - 1 = v` and the first value `MAX + gap + 1 = gap`
// (wrapping), which is the `None` rule without a branch. The inverse of
// `gaps` is [`delta::simd::values`].

/// The four gaps of the group `v` following `prev`: each lane less the lane
/// before it, the first lane's predecessor being `prev`'s last, less one.
#[inline]
#[target_feature(enable = "sse2")]
fn gaps(prev: V, v: V) -> V {
    let before = _mm_or_si128(_mm_slli_si128::<4>(v), _mm_srli_si128::<12>(prev));
    _mm_sub_epi32(_mm_sub_epi32(v, before), _mm_set1_epi32(1))
}
