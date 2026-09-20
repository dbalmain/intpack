//! The scalar kernel: the specification of the packed layout, in direct form.
//! Every element is placed (or read) independently from its bit position, so
//! there is no carried state to get wrong; `simd` re-derives the same bytes
//! incrementally and is tested against this.
//!
//! Element `i` lives in lane `i % 4` at bit `(i / 4) * width` of that lane's
//! LSB-first stream; word `k` of lane `j` is the little-endian `u32` at
//! `packed[(4 * k + j) * 4 ..]`. Values must fit in `width` bits.

use super::{BLOCK, mask};
use crate::delta;

/// Flat word index of word `k` of the lane holding element `i`.
fn word(i: usize, k: u32) -> usize {
    4 * k as usize + i % 4
}

/// Bit offset of element `i` within its lane's stream, as `(word, shift)`.
fn position(i: usize, width: u32) -> (u32, u32) {
    let bit = (i / 4) as u32 * width;
    (bit / 32, bit % 32)
}

/// `out` must be `packed_len(width)` bytes; it is overwritten.
pub fn pack(width: u32, block: &[u32; BLOCK], out: &mut [u8]) {
    out.fill(0);
    if width == 0 {
        return;
    }
    let words = out.as_chunks_mut::<4>().0;
    let mut or_into =
        |w: usize, bits: u32| words[w] = (u32::from_le_bytes(words[w]) | bits).to_le_bytes();
    for (i, &v) in block.iter().enumerate() {
        let (k, shift) = position(i, width);
        or_into(word(i, k), v << shift);
        // A value straddles two words only when `shift > 0`, so `32 - shift`
        // is a legal shift count.
        if shift + width > 32 {
            or_into(word(i, k + 1), v >> (32 - shift));
        }
    }
}

/// `packed` must be `packed_len(width)` bytes.
pub fn unpack(width: u32, packed: &[u8], out: &mut [u32; BLOCK]) {
    if width == 0 {
        out.fill(0);
        return;
    }
    let words = packed.as_chunks::<4>().0;
    let word_at = |w: usize| u32::from_le_bytes(words[w]);
    for (i, v) in out.iter_mut().enumerate() {
        let (k, shift) = position(i, width);
        let mut bits = word_at(word(i, k)) >> shift;
        if shift + width > 32 {
            bits |= word_at(word(i, k + 1)) << (32 - shift);
        }
        *v = bits & mask(width);
    }
}

// ── sorted blocks: the delta convention, then the same packing ──

/// The gaps of a sorted block, per [`delta::gap`], `base` being the element
/// before the block.
fn gaps(base: Option<u32>, block: &[u32; BLOCK]) -> [u32; BLOCK] {
    let mut gaps = [0; BLOCK];
    let mut prev = base;
    for (gap, &v) in gaps.iter_mut().zip(block) {
        *gap = delta::gap(prev, v);
        prev = Some(v);
    }
    gaps
}

pub fn width_sorted(base: Option<u32>, block: &[u32; BLOCK]) -> u32 {
    super::width(&gaps(base, block))
}

pub fn pack_sorted(width: u32, base: Option<u32>, block: &[u32; BLOCK], out: &mut [u8]) {
    pack(width, &gaps(base, block), out);
}

pub fn unpack_sorted(width: u32, base: Option<u32>, packed: &[u8], out: &mut [u32; BLOCK]) {
    unpack(width, packed, out);
    delta::decode(base, out);
}
