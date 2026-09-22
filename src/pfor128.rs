//! Patched bitpacked 128-blocks — Lucene's at most seven inline exceptions —
//! with a varint tail, and the container over the [`bitpack`] primitive.
//!
//! Layout: every full block of 128 values starts with a token
//! `(exceptions << 5) | width`, then — unsorted blocks only — the block
//! minimum as a LEB128 varint, then up to seven `[index: u8][high: u8]`
//! exception pairs, then `bitpack::packed_len(width)` packed low bits.
//! Sorted blocks store gaps and have no minimum: `gap - 1` is almost always
//! already based at 0, and a minimum would cost the fused `unpack_sorted`
//! path. Unsorted blocks always write that varint, one byte when the minimum
//! is 0; the token's eight bits are all `exceptions << 5 | width`, so there
//! is no flag to omit it. Packed lows are the gaps, or the values with the
//! minimum subtracted. A sorted block with no exceptions is byte-identical
//! to the `bitpacking` crate's `BitPacker4x` block with a width byte in
//! front. Token `0xff` escapes to 128 raw little-endian `u32`s of the
//! original values and carries no minimum; the escape is decided on the
//! values that would have been packed, so a block whose originals sit at or
//! above `2^31` can still pack once the minimum is subtracted. Seven
//! exceptions at width 30 would be token `0xfe`; that block goes raw too,
//! and `0xfe` marks a **bitset** block, sorted lists only: the gap-1 of the
//! block's first id from its base as a varint, a body length byte, then
//! that many bytes in which bit `i` is set iff `first + i` is in the list
//! (bit 0 always is). A block is a bitset when that is strictly fewer bytes
//! than patching it — dense stretches whose eighth-largest gap still forces
//! a wide packed block — and never when its span exceeds 2040 ids; a tie
//! stays packed, and a width-0 run is one byte and always wins. `next_geq`
//! on a bitset is a word operation with no decode. The `n % 128` values
//! after the last full block are LEB128 varints. Sorted gaps follow
//! [`delta`]: the first block's first value as-is, each later value `gap - 1`
//! from the one before it, across block boundaries, so a block's delta base
//! is the previous block's last value and the tail continues from the last
//! block. `n` is the caller's to keep.
//!
//! No skip data: the cursor decodes blocks in order and scans, so a seek to
//! the far end of a list unpacks everything before it. [`pfor128skip`] adds a
//! table in front of this exact payload and reuses the `pub(crate)` block and
//! tail helpers here.
//!
//! [`bitpack`]: crate::bitpack
//! [`delta`]: crate::delta
//! [`pfor128skip`]: crate::pfor128skip

use crate::bitpack::{self, BLOCK};
use crate::{Cursor, bits, delta, vbyte};

// ── blocks ──

const MAX_EXCEPTIONS: usize = 7;
const RAW_TOKEN: u8 = u8::MAX;
const BITSET_TOKEN: u8 = u8::MAX - 1;
/// Widest `last - first` a bitset block may cover: its body is at most 255 bytes.
const BITSET_MAX_SPAN: u32 = 255 * 8 - 1;

const fn low_mask(width: u32) -> u32 {
    ((1u64 << width) - 1) as u32
}

const fn bits_required(value: u32) -> u32 {
    32 - value.leading_zeros()
}

/// Lucene's fixed exception rule: the eighth-largest value sets the floor,
/// and an exception high part may occupy at most one byte.
fn patched_width(block: &[u32; BLOCK]) -> (u32, usize) {
    let max_bits = bitpack::width(block);
    let mut copy = *block;
    // Eighth-largest: the element at `BLOCK - 8` once the copy is partitioned.
    let (_, eighth, _) = copy.select_nth_unstable(BLOCK - 8);
    let width = bits_required(*eighth).max(max_bits.saturating_sub(8));
    let mask = low_mask(width);
    let exceptions = block.iter().filter(|&&v| v > mask).count();
    debug_assert!(exceptions <= MAX_EXCEPTIONS);
    (width, exceptions)
}

/// Tokens `0xff` and `0xfe` are taken, so a block that would use either
/// goes raw: seven exceptions at width 30 or 31 are gaps near `2^30`.
fn raw_escape(width: u32, exceptions: usize) -> bool {
    width == 32 || (width >= 30 && exceptions == MAX_EXCEPTIONS)
}

/// Bytes of a patched block: token, exception pairs, packed lows.
fn patched_bytes(width: u32, exceptions: usize) -> usize {
    1 + 2 * exceptions + bitpack::packed_len(width)
}

/// Body bytes of a bitset over `block`, `None` when it spans too far.
fn bitset_body_len(block: &[u32; BLOCK]) -> Option<usize> {
    let span = block[BLOCK - 1] - block[0];
    (span <= BITSET_MAX_SPAN).then(|| span as usize / 8 + 1)
}

/// Bytes of a bitset block: token, `first` varint, body length, body.
fn bitset_bytes(first: u32, body_len: usize) -> usize {
    1 + bits_required(first).max(1).div_ceil(7) as usize + 1 + body_len
}

/// Token, `first`, body length, then the body: bit `v - block[0]` set for
/// each `v` in `block`. Out of line: inlined into the block encoder it cost
/// the packed path 9% of encode throughput on lists with no bitset at all.
#[inline(never)]
fn write_bitset(first: u32, body_len: usize, block: &[u32; BLOCK], out: &mut Vec<u8>) {
    out.push(BITSET_TOKEN);
    vbyte::put(first, out);
    out.push(body_len as u8);
    let start = out.len();
    out.resize(start + body_len, 0);
    let body = &mut out[start..];
    for &v in block {
        let bit = (v - block[0]) as usize;
        body[bit / 8] |= 1 << (bit % 8);
    }
}

fn write_raw(block: &[u32; BLOCK], out: &mut Vec<u8>) {
    out.push(RAW_TOKEN);
    for &v in block {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// Exception pairs, then the packed low bits of `block`. The token is already written.
fn write_patched(block: &[u32; BLOCK], width: u32, out: &mut Vec<u8>) {
    let mut lows = *block;
    let mask = low_mask(width);
    for (i, v) in lows.iter_mut().enumerate() {
        if *v > mask {
            out.extend_from_slice(&[i as u8, (*v >> width) as u8]);
            *v &= mask;
        }
    }
    let start = out.len();
    out.resize(start + bitpack::packed_len(width), 0);
    bitpack::pack(width, &lows, &mut out[start..]);
}

fn block_min(block: &[u32; BLOCK]) -> u32 {
    block.iter().copied().fold(u32::MAX, u32::min)
}

/// Encode one full block of a sorted list, gaps relative to `base` (the
/// previous block's last value, `None` for the first block). Returns the
/// block's last value, the next block's base.
///
/// A bitset is chosen only when it is strictly smaller than the patched
/// block, so a dense block that ties stays packed on the fused
/// `unpack_sorted` path; a width-0 run is one byte and always wins.
pub(crate) fn encode_block_sorted(
    base: Option<u32>,
    block: &[u32; BLOCK],
    out: &mut Vec<u8>,
) -> u32 {
    let mut gaps = [0; BLOCK];
    let mut prev = base;
    for (&v, gap) in block.iter().zip(&mut gaps) {
        *gap = delta::gap(prev, v);
        prev = Some(v);
    }
    let (width, exceptions) = patched_width(&gaps);
    if raw_escape(width, exceptions) {
        write_raw(&gaps, out);
    } else if let Some(body_len) = bitset_body_len(block)
        && bitset_bytes(gaps[0], body_len) < patched_bytes(width, exceptions)
    {
        write_bitset(gaps[0], body_len, block, out);
    } else {
        out.push(((exceptions as u8) << 5) | width as u8);
        write_patched(&gaps, width, out);
    }
    block[BLOCK - 1]
}

/// Encode one full unsorted block: frame minimum, then the patched
/// minimum-subtracted values. A raw escape stores the original values and
/// no minimum — the escape is decided after the subtraction, so a block that
/// would have been raw can pack.
pub(crate) fn encode_block(block: &[u32; BLOCK], out: &mut Vec<u8>) {
    let min = block_min(block);
    let mut framed = *block;
    for v in &mut framed {
        *v -= min;
    }
    let (width, exceptions) = patched_width(&framed);
    if raw_escape(width, exceptions) {
        write_raw(block, out);
        return;
    }
    out.push(((exceptions as u8) << 5) | width as u8);
    vbyte::put(min, out);
    write_patched(&framed, width, out);
}

/// Decode the full block whose token byte is at `buf[*byte]`, advancing
/// `*byte` past it. Returns the block's last value.
pub(crate) fn decode_block_sorted(
    base: Option<u32>,
    buf: &[u8],
    byte: &mut usize,
    out: &mut [u32; BLOCK],
) -> u32 {
    let token = buf[*byte];
    if token == RAW_TOKEN {
        take_raw(buf, byte, out);
        delta::decode(base, out);
    } else if token == BITSET_TOKEN {
        Bitset::take(base, buf, byte).decode(out);
    } else {
        let exceptions = usize::from(token >> 5);
        let width = u32::from(token & 0x1f);
        if exceptions == 0 {
            let (_, packed) = take_patched(buf, byte);
            bitpack::unpack_sorted(width, base, packed, out);
        } else {
            let (patches, packed) = take_patched(buf, byte);
            bitpack::unpack(width, packed, out);
            apply_exceptions(width, patches, out);
            delta::decode(base, out);
        }
    }
    out[BLOCK - 1]
}

/// Unsorted [`decode_block_sorted`]: token, frame minimum, then the same
/// pairs and packed lows. The minimum is added back unless it is 0.
pub(crate) fn decode_block(buf: &[u8], byte: &mut usize, out: &mut [u32; BLOCK]) {
    let token = buf[*byte];
    if token == RAW_TOKEN {
        take_raw(buf, byte, out);
        return;
    }
    *byte += 1;
    let min = vbyte::take(buf, byte);
    let width = u32::from(token & 0x1f);
    let exceptions = usize::from(token >> 5);
    let patches = &buf[*byte..*byte + exceptions * 2];
    *byte += exceptions * 2;
    let packed_len = bitpack::packed_len(width);
    let packed = &buf[*byte..*byte + packed_len];
    *byte += packed_len;
    if exceptions == 0 {
        bitpack::unpack_add(width, min, packed, out);
    } else {
        bitpack::unpack(width, packed, out);
        apply_exceptions(width, patches, out);
        if min != 0 {
            for v in out {
                *v += min;
            }
        }
    }
}

/// Exception pairs and packed bytes at `buf[*byte]`, advancing past the block.
fn take_patched<'a>(buf: &'a [u8], byte: &mut usize) -> (&'a [u8], &'a [u8]) {
    let token = buf[*byte];
    let width = u32::from(token & 0x1f);
    let exception_bytes = usize::from(token >> 5) * 2;
    let patches = &buf[*byte + 1..*byte + 1 + exception_bytes];
    let packed_start = *byte + 1 + exception_bytes;
    let packed = &buf[packed_start..packed_start + bitpack::packed_len(width)];
    *byte = packed_start + packed.len();
    (patches, packed)
}

fn apply_exceptions(width: u32, patches: &[u8], out: &mut [u32; BLOCK]) {
    for pair in patches.as_chunks::<2>().0 {
        out[usize::from(pair[0])] |= u32::from(pair[1]) << width;
    }
}

fn take_raw(buf: &[u8], byte: &mut usize, out: &mut [u32; BLOCK]) {
    *byte += 1;
    for v in out {
        let mut bytes = [0; 4];
        bytes.copy_from_slice(&buf[*byte..*byte + 4]);
        *v = u32::from_le_bytes(bytes);
        *byte += 4;
    }
}

/// Bytes the sorted block at `buf[byte]` occupies, token included.
///
/// Unsorted blocks are longer by their minimum varint; [`block_len_unsorted`]
/// is that length. The two layouts share a token, so one function cannot
/// serve both.
pub(crate) fn block_len(buf: &[u8], byte: usize) -> usize {
    let token = buf[byte];
    if token == RAW_TOKEN {
        1 + BLOCK * size_of::<u32>()
    } else if token == BITSET_TOKEN {
        let mut pos = byte + 1;
        vbyte::take(buf, &mut pos);
        pos + 1 + usize::from(buf[pos]) - byte
    } else {
        patched_bytes(u32::from(token & 0x1f), usize::from(token >> 5))
    }
}

// ── bitset blocks ──

/// Whether the sorted block with this token is a bitset.
#[inline]
pub(crate) fn is_bitset(token: u8) -> bool {
    token == BITSET_TOKEN
}

/// A bitset block in place: bit `i` of `body`, little-endian, is set iff
/// `first + i` is in the list, and bit 0 always is. The body is read as
/// `u64` words on the fly; it is not aligned in `buf` and its last word is
/// padded with zeros.
#[derive(Clone, Copy)]
pub(crate) struct Bitset<'a> {
    /// The block's first id.
    pub(crate) first: u32,
    body: &'a [u8],
}

impl<'a> Bitset<'a> {
    /// The block at `buf[*byte]`, whose token is [`BITSET_TOKEN`], advancing
    /// `*byte` past it.
    pub(crate) fn take(base: Option<u32>, buf: &'a [u8], byte: &mut usize) -> Self {
        *byte += 1;
        let first = delta::value(base, vbyte::take(buf, byte));
        let len = usize::from(buf[*byte]);
        *byte += 1;
        let body = &buf[*byte..*byte + len];
        *byte += len;
        Self { first, body }
    }

    fn words(&self) -> usize {
        self.body.len().div_ceil(8)
    }

    #[inline]
    fn word(&self, i: usize) -> u64 {
        let rest = &self.body[i * 8..];
        match rest.first_chunk::<8>() {
            Some(word) => u64::from_le_bytes(*word),
            None => Self::tail_word(rest),
        }
    }

    /// The last word when the body is not a multiple of 8 bytes, zero
    /// padded. Kept out of line so [`Self::word`] stays a load and a branch
    /// at every inlined site; the readers on the hot path are size sensitive.
    #[cold]
    #[inline(never)]
    fn tail_word(rest: &[u8]) -> u64 {
        let mut word = [0; 8];
        word[..rest.len()].copy_from_slice(rest);
        u64::from_le_bytes(word)
    }

    /// The first set bit at or after bit `i`, which a cursor stepping one
    /// element at a time knows is set. Small enough to inline into `next()`,
    /// unlike [`Self::next_set`] with its skip count.
    #[inline]
    pub(crate) fn successor(&self, i: u32) -> u32 {
        let mut at = (i / 64) as usize;
        let mut word = self.word(at) & (u64::MAX << (i % 64));
        while word == 0 {
            at += 1;
            word = self.word(at);
        }
        at as u32 * 64 + word.trailing_zeros()
    }

    /// The first set bit at or after bit `i`, with the number of set bits in
    /// `from..bit` (`from <= i`) so a cursor can keep its element index;
    /// `None` when no bit at or after `i` is set.
    #[inline]
    pub(crate) fn next_set(&self, from: u32, i: u32) -> Option<(u32, u32)> {
        let words = self.words();
        let mut at = (i / 64) as usize;
        if at >= words {
            return None;
        }
        let mut wi = (from / 64) as usize;
        let mut word = self.word(wi) & (u64::MAX << (from % 64));
        let mut skipped = 0;
        while wi < at {
            skipped += word.count_ones();
            wi += 1;
            word = self.word(wi);
        }
        let from_i = u64::MAX << (i % 64);
        skipped += (word & !from_i).count_ones();
        word &= from_i;
        while word == 0 {
            at += 1;
            if at == words {
                return None;
            }
            word = self.word(at);
        }
        Some((at as u32 * 64 + word.trailing_zeros(), skipped))
    }

    /// The `j`th id of the block: popcount to its word, then select in it.
    pub(crate) fn select(&self, mut j: usize) -> u32 {
        for wi in 0..self.words() {
            let word = self.word(wi);
            let count = word.count_ones() as usize;
            if j < count {
                return self.first + (wi as u32 * 64 + bits::select_one(word, j));
            }
            j -= count;
        }
        unreachable!("a bitset block holds {BLOCK} ids")
    }

    /// The last id of the block: the highest set bit.
    pub(crate) fn last(&self) -> u32 {
        let top = self.words() - 1;
        self.first + (top as u32 * 64 + 63 - self.word(top).leading_zeros())
    }

    /// All 128 ids, in order. Each byte's set-bit positions come from a
    /// table and are written eight lanes at a time, so there is no
    /// per-bit dependency chain; the padded lanes are overwritten by the
    /// next byte, and the scratch absorbs the final overrun.
    fn decode(&self, out: &mut [u32; BLOCK]) {
        let mut scratch = [0; BLOCK + 8];
        let mut k = 0;
        for (i, &byte) in self.body.iter().enumerate() {
            let origin = self.first + i as u32 * 8;
            let lanes = &mut scratch[k..k + 8];
            for (lane, &bit) in lanes.iter_mut().zip(&BIT_POSITIONS[usize::from(byte)]) {
                *lane = origin + u32::from(bit);
            }
            k += byte.count_ones() as usize;
        }
        debug_assert_eq!(k, BLOCK);
        out.copy_from_slice(&scratch[..BLOCK]);
    }
}

/// For each byte value, the positions of its set bits, lowest first, padded
/// with zeros.
const BIT_POSITIONS: [[u8; 8]; 256] = {
    let mut table = [[0; 8]; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut lane = 0;
        let mut bit = 0;
        while bit < 8 {
            if byte >> bit & 1 == 1 {
                table[byte][lane] = bit as u8;
                lane += 1;
            }
            bit += 1;
        }
        byte += 1;
    }
    table
};

/// Bytes the unsorted block at `buf[byte]` occupies, token and frame minimum included.
pub(crate) fn block_len_unsorted(buf: &[u8], byte: usize) -> usize {
    let token = buf[byte];
    if token == RAW_TOKEN {
        1 + BLOCK * size_of::<u32>()
    } else {
        let mut pos = byte + 1;
        let _min = vbyte::take(buf, &mut pos);
        (pos - byte) + usize::from(token >> 5) * 2 + bitpack::packed_len(u32::from(token & 0x1f))
    }
}

// ── tail ──

/// Encode the sorted tail as gaps from `base`, the last full block's last
/// value (`None` when the list has no full block).
pub(crate) fn encode_tail_sorted(base: Option<u32>, tail: &[u32], out: &mut Vec<u8>) {
    let mut prev = base;
    for &v in tail {
        vbyte::put(delta::gap(prev, v), out);
        prev = Some(v);
    }
}

/// Decode `n` sorted tail values from `buf[*byte]`, appending to `out`.
pub(crate) fn decode_tail_sorted(
    base: Option<u32>,
    buf: &[u8],
    byte: &mut usize,
    n: usize,
    out: &mut Vec<u32>,
) {
    let mut prev = base;
    for _ in 0..n {
        let v = delta::value(prev, vbyte::take(buf, byte));
        out.push(v);
        prev = Some(v);
    }
}

pub(crate) fn encode_tail(tail: &[u32], out: &mut Vec<u8>) {
    for &v in tail {
        vbyte::put(v, out);
    }
}

pub(crate) fn decode_tail(buf: &[u8], byte: &mut usize, n: usize, out: &mut Vec<u32>) {
    for _ in 0..n {
        out.push(vbyte::take(buf, byte));
    }
}

// ── lists ──

/// Encode a strictly increasing list, appending to `out`.
pub fn encode_sorted(list: &[u32], out: &mut Vec<u8>) {
    let (blocks, tail) = list.as_chunks::<BLOCK>();
    let mut base = None;
    for block in blocks {
        base = Some(encode_block_sorted(base, block, out));
    }
    encode_tail_sorted(base, tail, out);
}

/// Decode the `n` values of a sorted list from `buf`, appending to `out`.
pub fn decode_sorted(n: usize, buf: &[u8], out: &mut Vec<u32>) {
    let start = out.len();
    out.resize(start + n / BLOCK * BLOCK, 0);
    let mut byte = 0;
    let mut base = None;
    for block in out[start..].as_chunks_mut::<BLOCK>().0 {
        base = Some(decode_block_sorted(base, buf, &mut byte, block));
    }
    decode_tail_sorted(base, buf, &mut byte, n % BLOCK, out);
}

/// Encode an arbitrary list, appending to `out`.
pub fn encode(list: &[u32], out: &mut Vec<u8>) {
    let (blocks, tail) = list.as_chunks::<BLOCK>();
    for block in blocks {
        encode_block(block, out);
    }
    encode_tail(tail, out);
}

/// Decode the `n` values of an unsorted list from `buf`, appending to `out`.
pub fn decode(n: usize, buf: &[u8], out: &mut Vec<u32>) {
    let start = out.len();
    out.resize(start + n / BLOCK * BLOCK, 0);
    let mut byte = 0;
    for block in out[start..].as_chunks_mut::<BLOCK>().0 {
        decode_block(buf, &mut byte, block);
    }
    decode_tail(buf, &mut byte, n % BLOCK, out);
}

// ── cursor ──

/// First index in `block[start..]` whose value is `>= target`.
///
/// The scalar loop is the reference. SSE2 compares four lanes at a time;
/// `_mm_cmpgt_epi32` is signed, so both sides are xored with `0x8000_0000`
/// and the unsigned order falls out of the signed compare. AVX-512 compares
/// sixteen, unsigned, straight to a mask. A leapfrog's hop is a few ids and
/// pays per compare-and-branch round, not per lane: at ∩ 1:10 the four-lane
/// scan cost twice Lucene's scalar loop and the sixteen-lane one is level.
#[inline]
pub(crate) fn scan_geq(block: &[u32; BLOCK], start: usize, target: u32) -> Option<usize> {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    // SAFETY: `scan_geq_avx512` requires AVX-512F, and this arm is compiled
    // only where that feature is enabled.
    unsafe {
        scan_geq_avx512(block, start, target)
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse2",
        not(target_feature = "avx512f")
    ))]
    // SAFETY: `scan_geq_sse2` requires SSE2, and this arm is compiled only
    // where that feature is enabled.
    unsafe {
        scan_geq_sse2(block, start, target)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
    scan_geq_scalar(block, start, target)
}

#[cfg(any(test, not(all(target_arch = "x86_64", target_feature = "sse2"))))]
fn scan_geq_scalar(block: &[u32; BLOCK], start: usize, target: u32) -> Option<usize> {
    (start..BLOCK).find(|&j| block[j] >= target)
}

#[cfg(any(test, all(target_arch = "x86_64", target_feature = "avx512f")))]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn scan_geq_avx512(block: &[u32; BLOCK], start: usize, target: u32) -> Option<usize> {
    use core::arch::x86_64::{_mm512_cmpge_epu32_mask, _mm512_loadu_si512, _mm512_set1_epi32};
    // SAFETY: each load is a 64-byte group inside `block` (`i` stays on a
    // 16-element boundary and stops at `BLOCK`).
    unsafe {
        let needle = _mm512_set1_epi32(target as i32);
        let mut i = start & !15;
        let mut skip = (start - i) as u32;
        while i < BLOCK {
            let hits =
                (_mm512_cmpge_epu32_mask(_mm512_loadu_si512(block.as_ptr().add(i).cast()), needle)
                    as u32
                    >> skip)
                    << skip;
            if hits != 0 {
                return Some(i + hits.trailing_zeros() as usize);
            }
            i += 16;
            skip = 0;
        }
        None
    }
}

#[cfg(any(
    test,
    all(
        target_arch = "x86_64",
        target_feature = "sse2",
        not(target_feature = "avx512f")
    )
))]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
fn scan_geq_sse2(block: &[u32; BLOCK], start: usize, target: u32) -> Option<usize> {
    use core::arch::x86_64::{
        _mm_castsi128_ps, _mm_cmpgt_epi32, _mm_loadu_si128, _mm_movemask_ps, _mm_set1_epi32,
        _mm_xor_si128,
    };
    // SAFETY: each load is a 16-byte group inside `block` (`i` stays on a
    // 4-element boundary and stops at `BLOCK`). The xor maps unsigned order
    // onto a signed compare; movemask reads the sign bit of each lane.
    unsafe {
        let bias = _mm_set1_epi32(i32::MIN);
        let needle = _mm_xor_si128(_mm_set1_epi32(target as i32), bias);
        let mut i = start & !3;
        // Only the first group has lanes before `start`. Later groups are whole.
        let mut skip = (start - i) as u32;
        while i < BLOCK {
            let below = _mm_cmpgt_epi32(
                needle,
                _mm_xor_si128(_mm_loadu_si128(block.as_ptr().add(i).cast()), bias),
            );
            // Bit `lane` set where the value is `>= target`.
            let hits =
                (((!_mm_movemask_ps(_mm_castsi128_ps(below))) & 0b1111) as u32 >> skip) << skip;
            if hits != 0 {
                return Some(i + hits.trailing_zeros() as usize);
            }
            i += 4;
            skip = 0;
        }
        None
    }
}

/// The block the scan cursor is in. Only `Values` fills
/// [`SortedCursor::block`].
#[derive(Clone, Copy)]
enum Entered<'a> {
    /// `block` holds this block's ids.
    Values,
    /// Width-0 block: element `j` is `first + j`.
    Run { first: u32 },
    /// Bitset block, answered by word operations on its bytes in place.
    Bits(Bitset<'a>),
}

/// Block-at-a-time cursor over a sorted list: unpacks each full block when
/// the scan enters it, then reads the tail varint by varint. A width-0 block
/// is 128 consecutive ids and is not unpacked; a bitset block is walked by
/// word, and passed over by reading its last word.
pub struct SortedCursor<'a> {
    buf: &'a [u8],
    n: usize,
    /// Elements consumed so far; the current element is `pos - 1`.
    pos: usize,
    /// Where the next block, or the next tail varint, starts in `buf`.
    byte: usize,
    block: [u32; BLOCK],
    entered: Entered<'a>,
    /// The current element, `None` before the first and after the last.
    cur: Option<u32>,
}

impl<'a> SortedCursor<'a> {
    /// Open the `n`-element list encoded in `buf`, positioned before its
    /// first element.
    pub fn new(n: usize, buf: &'a [u8]) -> Self {
        Self {
            buf,
            n,
            pos: 0,
            byte: 0,
            block: [0; BLOCK],
            entered: Entered::Values,
            cur: None,
        }
    }

    /// Decode the block at `pos`, or record a width-0 run or a bitset. A
    /// zero token is one byte: width 0 packs to nothing and there are no
    /// exceptions, and every gap is 0, so the ids are `first, first + 1, …`
    /// with `first` being `cur + 1` (or `0` at the head of the list).
    fn enter_block(&mut self) {
        let token = self.buf[self.byte];
        self.entered = if token == 0 {
            self.byte += 1;
            Entered::Run {
                first: delta::value(self.cur, 0),
            }
        } else if token == BITSET_TOKEN {
            Entered::Bits(Bitset::take(self.cur, self.buf, &mut self.byte))
        } else {
            decode_block_sorted(self.cur, self.buf, &mut self.byte, &mut self.block);
            Entered::Values
        };
    }

    /// Position on element `j` of block `b`, whose id is `value`.
    fn land(&mut self, b: usize, j: usize, value: u32) -> Option<u32> {
        self.pos = b * BLOCK + j + 1;
        self.cur = Some(value);
        self.cur
    }
}

/// `v`, as a value the optimiser must materialise here: an empty asm
/// with `v` as an in-out register costs no instruction and keeps LLVM from
/// hoisting what is computed from `v` out of the enclosing loop.
///
/// A bet on one compiler's register allocation, not part of the algorithm:
/// it is on the ledger in `src/toolchain.rs`, whose test fails when rustc
/// changes so the measurement behind it is repeated. Removing it must cost
/// nothing but speed.
#[inline(always)]
fn pin(mut v: u32) -> u32 {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: the asm is empty and only names a register operand.
    unsafe {
        core::arch::asm!("/* {0:e} */", inout(reg) v, options(pure, nomem, nostack, preserves_flags));
    }
    v
}

impl Cursor for SortedCursor<'_> {
    fn next_geq(&mut self, target: u32) -> Option<u32> {
        if let Some(cur) = self.cur
            && cur >= target
        {
            return Some(cur);
        }
        let full = self.n / BLOCK * BLOCK;
        while self.pos < full {
            let b = self.pos / BLOCK;
            let start = self.pos % BLOCK;
            if start == 0 {
                self.enter_block();
            }
            let last = match self.entered {
                Entered::Run { first } => {
                    let last = first + (BLOCK as u32 - 1);
                    if target <= last {
                        let j = if target <= first {
                            start
                        } else {
                            ((target - first) as usize).max(start)
                        };
                        return self.land(b, j, first + j as u32);
                    }
                    last
                }
                Entered::Bits(bits) => {
                    // Bits before `start` are already counted in `pos`.
                    let from = self.cur.map_or(0, |c| (c + 1).saturating_sub(bits.first));
                    let i = target.saturating_sub(bits.first).max(from);
                    if let Some((bit, skipped)) = bits.next_set(from, i) {
                        return self.land(b, start + skipped as usize, bits.first + bit);
                    }
                    bits.last()
                }
                Entered::Values => {
                    // Pin `target` to this iteration: LLVM otherwise hoists
                    // the scan's broadcast of it above the loop and spills the
                    // vector across `enter_block`, and every call then pays a
                    // store-forward of it on the way to the compare.
                    let target = pin(target);
                    if let Some(j) = scan_geq(&self.block, start, target) {
                        return self.land(b, j, self.block[j]);
                    }
                    self.block[BLOCK - 1]
                }
            };
            self.pos = (b + 1) * BLOCK;
            self.cur = Some(last);
        }
        while let Some(v) = self.next() {
            if v >= target {
                return Some(v);
            }
        }
        None
    }

    fn next(&mut self) -> Option<u32> {
        if self.pos == self.n {
            self.cur = None;
            return None;
        }
        // The current element is the delta base for whatever comes next: the
        // previous block's last value, or the previous tail value.
        let v = if self.pos < self.n / BLOCK * BLOCK {
            if self.pos.is_multiple_of(BLOCK) {
                self.enter_block();
            }
            let j = self.pos % BLOCK;
            match self.entered {
                Entered::Run { first } => first + j as u32,
                Entered::Values => self.block[j],
                Entered::Bits(bits) => {
                    let from = if j == 0 {
                        0
                    } else {
                        self.cur.map_or(0, |c| c + 1 - bits.first)
                    };
                    match bits.next_set(from, from) {
                        Some((bit, _)) => bits.first + bit,
                        None => unreachable!("a bitset block holds {BLOCK} ids"),
                    }
                }
            }
        } else {
            delta::value(self.cur, vbyte::take(self.buf, &mut self.byte))
        };
        self.pos += 1;
        self.cur = Some(v);
        self.cur
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use rand::prelude::*;

    use super::*;

    /// A strictly increasing list of `n` values with random gaps of up to
    /// `spread`.
    pub(crate) fn sorted_list(rng: &mut StdRng, n: usize, spread: u32) -> Vec<u32> {
        let mut v = rng.random_range(0..spread);
        (0..n)
            .map(|_| {
                let cur = v;
                v += rng.random_range(1..=spread);
                cur
            })
            .collect()
    }

    /// The sorted lists every container test round-trips: empty, one element,
    /// around the block boundary, a dense run (all gaps zero), a list ending
    /// at `u32::MAX`, and random.
    pub(crate) fn sorted_cases() -> Vec<Vec<u32>> {
        let mut rng = StdRng::seed_from_u64(7);
        let mut cases: Vec<Vec<u32>> = [0, 1, 127, 128, 129, 1000]
            .iter()
            .map(|&n| sorted_list(&mut rng, n, 5))
            .collect();
        cases.push((0..300).collect());
        cases.push((0..300).map(|i| u32::MAX - 299 + i).collect());
        cases.push(vec![0, u32::MAX]);
        cases.push(sorted_list(&mut rng, 5000, 1 << 20));
        cases.push(mixed_blocks());
        cases
    }

    /// Gap-1 values of a block a bitset beats: eight gaps of 8 put the
    /// packed width at 3 (49 bytes) while the span is 260 (33 body bytes).
    /// The block's first id is its base plus 1.
    pub(crate) fn dense_gaps() -> [u32; BLOCK] {
        std::array::from_fn(|i| match i % 15 {
            14 => 7,
            0 | 3 | 6 | 9 | 12 => 0,
            _ => 1,
        })
    }

    /// The 128 ids of `gaps` after `prev` (`None` at the head of a list).
    pub(crate) fn block_after(prev: Option<u32>, mut gaps: [u32; BLOCK]) -> Vec<u32> {
        delta::decode(prev, &mut gaps);
        gaps.to_vec()
    }

    /// A spiked gap so the block is not token `0`, strictly increasing.
    pub(crate) fn patched_after(prev: u32) -> Vec<u32> {
        let mut gaps = [0; BLOCK];
        gaps[64] = 86_000 - 1;
        block_after(Some(prev), gaps)
    }

    /// One block of every sorted kind — run, bitset, packed with exceptions,
    /// bitset with a two-byte `first`, plain packed — then a tail. A sorted
    /// list cannot produce a raw block: eight gaps of `2^29` overflow `u32`.
    pub(crate) fn mixed_blocks() -> Vec<u32> {
        let mut list: Vec<u32> = (0..BLOCK as u32).collect();
        list.extend(block_after(list.last().copied(), dense_gaps()));
        list.extend(patched_after(list[list.len() - 1]));
        let mut far = dense_gaps();
        far[0] = 299;
        list.extend(block_after(list.last().copied(), far));
        list.extend(block_after(list.last().copied(), [2; BLOCK]));
        let last = list[list.len() - 1];
        list.extend((1..=5).map(|i| last + i * 3));
        list
    }

    /// Unsorted lists: the same shapes plus width-32 blocks.
    pub(crate) fn unsorted_cases() -> Vec<Vec<u32>> {
        let mut rng = StdRng::seed_from_u64(8);
        let mut cases: Vec<Vec<u32>> = [0, 1, 127, 128, 129, 1000]
            .iter()
            .map(|&n| (0..n).map(|_| rng.random_range(0..1000)).collect())
            .collect();
        cases.push(vec![0; 300]);
        cases.push((0..300).map(|_| rng.random()).collect());
        cases.push(vec![u32::MAX, 0, u32::MAX, 1]);
        cases
    }

    /// Drive `cursor` with the bench's semantics: seek to every element via
    /// the gap before it, re-seek to the element itself (must not move),
    /// then exhaust.
    pub(crate) fn check_cursor(list: &[u32], mut cursor: impl Cursor) {
        for (i, &v) in list.iter().enumerate() {
            let target = if i > 0 && list[i - 1] + 1 < v {
                list[i - 1] + 1
            } else {
                v
            };
            assert_eq!(cursor.next_geq(target), Some(v), "next_geq({target})");
            assert_eq!(cursor.next_geq(v), Some(v), "re-seek to {v} must stay put");
        }
        assert_eq!(
            cursor.next_geq(u32::MAX),
            list.last().copied().filter(|&l| l == u32::MAX)
        );
        assert_eq!(cursor.next(), None);
        assert_eq!(cursor.next(), None, "exhaustion is sticky");
    }

    #[test]
    fn sorted_roundtrip() {
        for list in sorted_cases() {
            let mut buf = Vec::new();
            encode_sorted(&list, &mut buf);
            let mut back = vec![7];
            decode_sorted(list.len(), &buf, &mut back);
            assert_eq!(back[0], 7, "decode appends");
            assert_eq!(back[1..], list, "n = {}", list.len());
        }
    }

    #[test]
    fn unsorted_roundtrip() {
        for list in unsorted_cases() {
            let mut buf = Vec::new();
            encode(&list, &mut buf);
            let mut back = Vec::new();
            decode(list.len(), &buf, &mut back);
            assert_eq!(back, list, "n = {}", list.len());
        }
    }

    #[test]
    fn dense_blocks_pack_to_one_byte() {
        let list: Vec<u32> = (0..256).collect();
        let mut buf = Vec::new();
        encode_sorted(&list, &mut buf);
        assert_eq!(buf, [0, 0]);
    }

    fn values_from_gaps(gaps: [u32; BLOCK]) -> Vec<u32> {
        block_after(None, gaps)
    }

    #[test]
    fn dense_block_is_a_bitset() {
        let list = values_from_gaps(dense_gaps());
        assert_eq!(list[BLOCK - 1] - list[0], 260, "span");
        let mut buf = Vec::new();
        encode_sorted(&list, &mut buf);
        // Token, `first` (gap-1 of id 0 from the head: one byte), 33 body bytes.
        assert_eq!(&buf[..3], &[BITSET_TOKEN, 0, 33]);
        assert_eq!(buf.len(), 1 + 1 + 1 + 33);
        assert_eq!(buf[3] & 1, 1, "bit 0 is always set");
        let mut back = Vec::new();
        decode_sorted(list.len(), &buf, &mut back);
        assert_eq!(back, list);

        // The same ids with the last one moved past the span cap: packed,
        // one width-4 exception, not a bitset.
        let mut gaps = dense_gaps();
        gaps[BLOCK - 1] = 3000;
        let list = values_from_gaps(gaps);
        let mut buf = Vec::new();
        encode_sorted(&list, &mut buf);
        assert_eq!(buf[0], (1 << 5) | 4);
        assert_eq!(buf.len(), 1 + 2 + 4 * 16);
        let mut back = Vec::new();
        decode_sorted(list.len(), &buf, &mut back);
        assert_eq!(back, list);
    }

    #[test]
    fn tie_stays_packed_and_one_id_closer_is_a_bitset() {
        // Eight gaps of 3, `twos` gaps of 2, the rest 1: width 2 with no
        // exceptions packs to 33 bytes, and the bitset is 3 bytes plus a
        // body of `span / 8 + 1`.
        let block = |twos: usize| {
            let mut gaps = [0; BLOCK];
            gaps[1..9].fill(2);
            gaps[9..9 + twos].fill(1);
            values_from_gaps(gaps)
        };
        let tie = block(90);
        assert_eq!(tie[BLOCK - 1] - tie[0], 233);
        let mut buf = Vec::new();
        encode_sorted(&tie, &mut buf);
        assert_eq!(buf[0], 2, "width 2, no exceptions");
        assert_eq!(buf.len(), 33);

        let closer = block(88);
        assert_eq!(closer[BLOCK - 1] - closer[0], 231);
        let mut buf = Vec::new();
        encode_sorted(&closer, &mut buf);
        assert_eq!(&buf[..3], &[BITSET_TOKEN, 0, 29]);
        assert_eq!(buf.len(), 32);
        let mut back = Vec::new();
        decode_sorted(closer.len(), &buf, &mut back);
        assert_eq!(back, closer);
    }

    #[test]
    fn bitset_first_is_the_gap_from_the_base() {
        let list = mixed_blocks();
        let mut buf = Vec::new();
        encode_sorted(&list, &mut buf);
        assert_eq!(buf[0], 0, "block 0 is a run");
        // Block 1 follows id 127 at 128: gap-1 is 0. Block 3's first id is
        // 300 past the end of block 2: a two-byte varint.
        assert_eq!(&buf[1..4], &[BITSET_TOKEN, 0, 33]);
        let b2 = 1 + block_len(&buf, 1);
        assert_ne!(buf[b2] >> 5, 0, "block 2 has exceptions");
        let b3 = b2 + block_len(&buf, b2);
        assert_eq!(&buf[b3..b3 + 4], &[BITSET_TOKEN, 0xab, 0x02, 33]);
        let b4 = b3 + block_len(&buf, b3);
        assert_eq!(buf[b4], 2, "block 4 is plain packed");
        assert_eq!(b4 + block_len(&buf, b4) + 5, buf.len(), "five tail bytes");
        let mut back = Vec::new();
        decode_sorted(list.len(), &buf, &mut back);
        assert_eq!(back, list);
    }

    #[test]
    fn seven_width_30_exceptions_escape_to_raw() {
        // `0xfe` would be the token of this block; it is the bitset token.
        let mut block = [1 << 29; BLOCK];
        block[..7].fill(u32::MAX);
        block[7] = 0;
        let buf = check_unsorted_block(block);
        assert_eq!(buf[0], RAW_TOKEN);
    }

    fn check_unsorted_block(block: [u32; BLOCK]) -> Vec<u8> {
        let mut buf = Vec::new();
        encode(&block, &mut buf);
        let mut back = Vec::new();
        decode(BLOCK, &buf, &mut back);
        assert_eq!(back, block);
        buf
    }

    #[test]
    fn one_outlier_uses_a_nine_bit_patched_block() {
        let mut gaps = [511; BLOCK];
        gaps[64] = 86_000;
        let mut list = values_from_gaps(gaps);
        list.push(list[BLOCK - 1] + 1);
        let mut buf = Vec::new();
        encode_sorted(&list, &mut buf);
        assert_eq!(buf.len(), 1 + 2 + 9 * 16 + 1);
        // Token, then the exception pair. A frame minimum would sit between them.
        assert_eq!(&buf[..3], &[(1 << 5) | 9, 64, 167]);
        let mut back = Vec::new();
        decode_sorted(list.len(), &buf, &mut back);
        assert_eq!(back, list);
    }

    #[test]
    fn exceptions_at_block_edges() {
        for index in [0, BLOCK - 1] {
            let mut block = [511; BLOCK];
            block[index] = 86_000;
            let buf = check_unsorted_block(block);
            // Minimum 511 is `0xff 0x03`. The stored high byte is
            // `(86_000 - 511) >> 9` = 166, not the unsubtracted 167.
            assert_eq!(&buf[..5], &[(1 << 5) | 9, 0xff, 0x03, index as u8, 166]);

            let list = values_from_gaps(block);
            let mut buf = Vec::new();
            encode_sorted(&list, &mut buf);
            assert_eq!(&buf[..3], &[(1 << 5) | 9, index as u8, 167]);
            let mut back = Vec::new();
            decode_sorted(list.len(), &buf, &mut back);
            assert_eq!(back, list);
        }
    }

    #[test]
    fn exception_high_byte_can_be_255() {
        // Subtracted outlier is `255 << 16`; the minimum is written in front of it.
        let min = 5u32;
        let mut block = [min; BLOCK];
        block[37] = min + (255 << 16);
        let buf = check_unsorted_block(block);
        assert_eq!(&buf[..4], &[(1 << 5) | 16, min as u8, 37, 255]);
    }

    #[test]
    fn seven_exceptions_fit_the_token() {
        let mut block = [511; BLOCK];
        block[..7].fill(86_000);
        let buf = check_unsorted_block(block);
        assert_eq!(buf[0], (7 << 5) | 9);
        assert_eq!(buf.len(), 1 + 2 + 7 * 2 + 9 * 16);
    }

    #[test]
    fn eight_equal_outliers_raise_the_width() {
        let mut block = [511; BLOCK];
        block[..8].fill(86_000);
        let buf = check_unsorted_block(block);
        assert_eq!(&buf[..3], &[17, 0xff, 0x03]);
        assert_eq!(buf.len(), 1 + 2 + 17 * 16);
    }

    #[test]
    fn equal_block_is_a_token_and_the_minimum() {
        assert_eq!(check_unsorted_block([1000; BLOCK]), [0x00, 0xe8, 0x07]);
        assert_eq!(check_unsorted_block([0; BLOCK]), [0x00, 0x00]);
    }

    #[test]
    fn frame_minimum_shrinks_the_width() {
        let block = std::array::from_fn(|i| 5000 + (i as u32 % 8));
        let buf = check_unsorted_block(block);
        // 5000 needs 13 bits; the frame is 0..7, so the token is width 3.
        assert_eq!(buf[0], 3);
        assert_eq!(buf.len(), 1 + 2 + bitpack::packed_len(3));
    }

    #[test]
    fn values_above_2_31_pack_when_the_spread_is_small() {
        let base = 1 << 31;
        let block = std::array::from_fn(|i| base + (i as u32 % 8));
        let buf = check_unsorted_block(block);
        assert_eq!(buf[0], 3, "not a raw escape");
        assert_eq!(&buf[1..6], &[0x80, 0x80, 0x80, 0x80, 0x08]);
    }

    #[test]
    fn raw_escape_for_width_32() {
        let mut block = [0; BLOCK];
        block[..8].fill(u32::MAX);
        let buf = check_unsorted_block(block);
        assert_eq!(buf[0], RAW_TOKEN);
        assert_eq!(buf.len(), 1 + BLOCK * 4);
    }

    #[test]
    fn raw_escape_for_seven_width_31_exceptions() {
        // A zero keeps the minimum at 0, so the range still takes the escape.
        let mut block = [1 << 30; BLOCK];
        block[..7].fill(u32::MAX);
        block[7] = 0;
        let buf = check_unsorted_block(block);
        assert_eq!(buf[0], RAW_TOKEN);
        assert_eq!(buf.len(), 1 + BLOCK * 4);
    }

    #[test]
    fn frame_minimum_packs_a_would_be_raw_block() {
        let mut block = [1 << 30; BLOCK];
        block[..7].fill(u32::MAX);
        let buf = check_unsorted_block(block);
        assert_ne!(buf[0], RAW_TOKEN);
        assert!(buf.len() < 1 + BLOCK * 4);
    }

    #[test]
    fn one_top_of_range_value_still_packs() {
        // `u32::MAX - 1` needs 32 bits, but only one value does, so the
        // exception rule packs it at width 24. Raw needs eight such values.
        let mut block = [1; BLOCK];
        block[0] = u32::MAX;
        let buf = check_unsorted_block(block);
        assert_eq!(buf[0], (1 << 5) | 24);
    }

    #[test]
    fn raw_when_subtraction_still_spans_the_range() {
        // Eight values still need 32 bits after subtracting the floor of 1,
        // so the block stays raw: original values, no minimum varint.
        let mut block = [1; BLOCK];
        block[..8].fill(u32::MAX);
        let buf = check_unsorted_block(block);
        assert_eq!(buf[0], RAW_TOKEN);
        assert_eq!(buf.len(), 1 + BLOCK * 4);
        assert_eq!(&buf[1..5], &u32::MAX.to_le_bytes());
    }

    #[test]
    fn exceptions_are_decided_after_subtraction() {
        let mut block = std::array::from_fn(|i| 100 + (i as u32 % 8));
        block[50] = 100 + 70_000;
        let buf = check_unsorted_block(block);
        assert_eq!(buf[0], (1 << 5) | 9);
        assert_eq!(&buf[1..4], &[100, 50, (70_000u32 >> 9) as u8]);
    }

    #[test]
    fn random_patched_blocks_across_widths() {
        let mut rng = StdRng::seed_from_u64(9);
        for width in 0..=30 {
            let lows = low_mask(width);
            let mut block = std::array::from_fn(|_| rng.random::<u32>() & lows);
            block[7] = lows;
            for v in &mut block[..7] {
                let high_bits = (32 - width).min(8);
                let high_mask = low_mask(high_bits);
                let high = rng.random_range(1..=high_mask);
                *v = (*v & lows) | (high << width);
            }
            let buf = check_unsorted_block(block);
            assert_ne!(buf[0] >> 5, 0, "width {width}");
        }
    }

    #[test]
    fn random_patched_blocks_with_a_frame_offset() {
        let mut rng = StdRng::seed_from_u64(12);
        for width in 0..=20 {
            let lows = low_mask(width);
            let mut block = std::array::from_fn(|_| rng.random::<u32>() & lows);
            block[7] = lows;
            for v in &mut block[..7] {
                let high_bits = (32 - width).min(8);
                let high_mask = low_mask(high_bits);
                let high = rng.random_range(1..=high_mask);
                *v = (*v & lows) | (high << width);
            }
            let max = block.iter().copied().fold(0, u32::max);
            let offset = rng.random_range(0..=u32::MAX - max);
            for v in &mut block {
                *v += offset;
            }
            check_unsorted_block(block);
        }
    }

    #[test]
    fn random_sorted_patched_blocks_across_widths() {
        let mut rng = StdRng::seed_from_u64(10);
        for width in 0..=16 {
            let lows = low_mask(width);
            let mut gaps = std::array::from_fn(|_| rng.random::<u32>() & lows);
            gaps[7] = lows;
            for gap in &mut gaps[..7] {
                let high = rng.random_range(1..=255);
                *gap = (*gap & lows) | (high << width);
            }
            let list = values_from_gaps(gaps);
            let mut buf = Vec::new();
            encode_sorted(&list, &mut buf);
            assert_ne!(buf[0] >> 5, 0, "width {width}");
            let mut back = Vec::new();
            decode_sorted(list.len(), &buf, &mut back);
            assert_eq!(back, list, "width {width}");
        }
    }

    type Scan = fn(&[u32; BLOCK], usize, u32) -> Option<usize>;

    /// Every `scan_geq` arm the host can execute, whichever one the crate
    /// was compiled to dispatch: the SIMD arms are compiled for tests on any
    /// x86_64 and run under runtime feature detection, so a baseline
    /// `cargo test` on an AVX-512 host covers all three.
    fn scan_arms() -> Vec<(&'static str, Scan)> {
        let mut arms: Vec<(&str, Scan)> = vec![("dispatched", scan_geq)];
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: every x86_64 target has SSE2.
            arms.push(("sse2", |b, s, t| unsafe { scan_geq_sse2(b, s, t) }));
            if std::arch::is_x86_feature_detected!("avx512f") {
                // SAFETY: AVX-512F was just detected on this host.
                arms.push(("avx512", |b, s, t| unsafe { scan_geq_avx512(b, s, t) }));
            }
        }
        arms
    }

    #[test]
    fn scan_agrees_on_random_blocks() {
        let mut rng = StdRng::seed_from_u64(11);
        let mut blocks = vec![
            [0u32; BLOCK],
            [u32::MAX; BLOCK],
            std::array::from_fn(|i| i as u32),
            std::array::from_fn(|i| u32::MAX - i as u32),
        ];
        for _ in 0..32 {
            blocks.push(std::array::from_fn(|_| rng.random()));
        }
        let targets = [0, 1, 2, 0x7fff_ffff, 0x8000_0000, u32::MAX, u32::MAX - 1];
        for (arm, scan) in scan_arms() {
            for block in &blocks {
                for &target in &targets {
                    for start in [0, 1, 2, 3, 4, 5, 15, 16, 17, 63, 64, 127, 128] {
                        assert_eq!(
                            scan(block, start, target),
                            scan_geq_scalar(block, start, target),
                            "{arm}: start {start} target {target:#x}"
                        );
                    }
                }
                // Targets taken from the block, including from the lanes a
                // group would otherwise mask off.
                for index in [0, 1, 3, 4, 15, 16, 127] {
                    let target = block[index];
                    for start in 0..=index {
                        assert_eq!(
                            scan(block, start, target),
                            scan_geq_scalar(block, start, target),
                            "{arm}: value at {index}, start {start}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn cursor_lands_on_a_patched_element() {
        let mut gaps = [0; BLOCK];
        gaps[64] = 86_000;
        let list = values_from_gaps(gaps);
        let mut buf = Vec::new();
        encode_sorted(&list, &mut buf);
        let mut cursor = SortedCursor::new(list.len(), &buf);
        assert_eq!(cursor.next_geq(list[63] + 1), Some(list[64]));
        assert_eq!(cursor.next_geq(list[64]), Some(list[64]));
    }

    #[test]
    fn cursor() {
        for list in sorted_cases() {
            let mut buf = Vec::new();
            encode_sorted(&list, &mut buf);
            check_cursor(&list, SortedCursor::new(list.len(), &buf));
            let mut cursor = SortedCursor::new(list.len(), &buf);
            for &v in &list {
                assert_eq!(cursor.next(), Some(v));
            }
            assert_eq!(cursor.next(), None);
        }
    }
}
