//! Bitpacked 128-blocks with a varint tail — the Lucene/Tantivy posting-list
//! shape, and the container over the [`bitpack`] primitive.
//!
//! Layout: every full block of 128 values starts with a token
//! `(exceptions << 5) | width`, then up to seven `[index: u8][high: u8]`
//! exception pairs, then `bitpack::packed_len(width)` packed low bits. Token
//! `0xff` escapes to 128 raw little-endian `u32`s. The `n % 128` values after
//! the last full block are LEB128 varints. Sorted lists store gaps per [`delta`]:
//! the first block's first value as-is, each later value `gap - 1` from the
//! one before it, across block boundaries, so a block's delta base is the
//! previous block's last value and the tail continues from the last block.
//! Unsorted lists store values as-is. `n` is the caller's to keep.
//!
//! No skip data: the cursor decodes blocks in order and scans, so a seek to
//! the far end of a list unpacks everything before it. [`bp128skip`] adds a
//! table in front of this exact payload and reuses the `pub(crate)` block and
//! tail helpers here.
//!
//! [`bitpack`]: crate::bitpack
//! [`delta`]: crate::delta
//! [`bp128skip`]: crate::bp128skip

use crate::bitpack::{self, BLOCK};
use crate::{Cursor, delta, vbyte};

// ── blocks ──

const MAX_EXCEPTIONS: usize = 7;
const RAW_TOKEN: u8 = u8::MAX;

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
    let mut ordered = *block;
    ordered.sort_unstable();
    let width = bits_required(ordered[BLOCK - 8]).max(max_bits.saturating_sub(8));
    let mask = low_mask(width);
    let exceptions = block.iter().filter(|&&v| v > mask).count();
    debug_assert!(exceptions <= MAX_EXCEPTIONS);
    (width, exceptions)
}

fn encode_values(block: &[u32; BLOCK], out: &mut Vec<u8>) {
    let (width, exceptions) = patched_width(block);
    if width == 32 || (width == 31 && exceptions == MAX_EXCEPTIONS) {
        out.push(RAW_TOKEN);
        for &v in block {
            out.extend_from_slice(&v.to_le_bytes());
        }
        return;
    }

    out.push(((exceptions as u8) << 5) | width as u8);
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

/// Encode one full block of a sorted list, gaps relative to `base` (the
/// previous block's last value, `None` for the first block). Returns the
/// block's last value, the next block's base.
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
    encode_values(&gaps, out);
    block[BLOCK - 1]
}

/// Encode one full block of values as-is.
pub(crate) fn encode_block(block: &[u32; BLOCK], out: &mut Vec<u8>) {
    encode_values(block, out);
}

/// Decode the full block whose width byte is at `buf[*byte]`, advancing
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

/// Unsorted [`decode_block_sorted`].
pub(crate) fn decode_block(buf: &[u8], byte: &mut usize, out: &mut [u32; BLOCK]) {
    let token = buf[*byte];
    if token == RAW_TOKEN {
        take_raw(buf, byte, out);
    } else {
        let width = u32::from(token & 0x1f);
        let (patches, packed) = take_patched(buf, byte);
        bitpack::unpack(width, packed, out);
        apply_exceptions(width, patches, out);
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

/// Bytes the block at `buf[byte]` occupies, token included.
pub(crate) fn block_len(buf: &[u8], byte: usize) -> usize {
    let token = buf[byte];
    if token == RAW_TOKEN {
        1 + BLOCK * size_of::<u32>()
    } else {
        1 + usize::from(token >> 5) * 2 + bitpack::packed_len(u32::from(token & 0x1f))
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

/// Block-at-a-time cursor over a sorted list: unpacks each full block when
/// the scan enters it, then reads the tail varint by varint.
pub struct SortedCursor<'a> {
    buf: &'a [u8],
    n: usize,
    /// Elements consumed so far; the current element is `pos - 1`.
    pos: usize,
    /// Where the next block, or the next tail varint, starts in `buf`.
    byte: usize,
    block: [u32; BLOCK],
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
            cur: None,
        }
    }
}

impl Cursor for SortedCursor<'_> {
    fn next_geq(&mut self, target: u32) -> Option<u32> {
        if let Some(cur) = self.cur
            && cur >= target
        {
            return Some(cur);
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
                decode_block_sorted(self.cur, self.buf, &mut self.byte, &mut self.block);
            }
            self.block[self.pos % BLOCK]
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
        cases
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

    fn values_from_gaps(mut gaps: [u32; BLOCK]) -> Vec<u32> {
        delta::decode(None, &mut gaps);
        gaps.to_vec()
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
        assert_eq!(buf[0], (1 << 5) | 9);
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
            assert_eq!(&buf[..3], &[(1 << 5) | 9, index as u8, 167]);
        }
    }

    #[test]
    fn exception_high_byte_can_be_255() {
        let mut block = [1 << 23; BLOCK];
        block[37] = u32::MAX;
        let buf = check_unsorted_block(block);
        assert_eq!(&buf[..3], &[(1 << 5) | 24, 37, 255]);
    }

    #[test]
    fn seven_exceptions_fit_the_token() {
        let mut block = [511; BLOCK];
        block[..7].fill(86_000);
        let buf = check_unsorted_block(block);
        assert_eq!(buf[0], (7 << 5) | 9);
        assert_eq!(buf.len(), 1 + 7 * 2 + 9 * 16);
    }

    #[test]
    fn eight_equal_outliers_raise_the_width() {
        let mut block = [511; BLOCK];
        block[..8].fill(86_000);
        let buf = check_unsorted_block(block);
        assert_eq!(buf[0], 17);
        assert_eq!(buf.len(), 1 + 17 * 16);
    }

    #[test]
    fn zero_block_has_zero_token() {
        assert_eq!(check_unsorted_block([0; BLOCK]), [0]);
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
        let mut block = [1 << 30; BLOCK];
        block[..7].fill(u32::MAX);
        let buf = check_unsorted_block(block);
        assert_eq!(buf[0], RAW_TOKEN);
        assert_eq!(buf.len(), 1 + BLOCK * 4);
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
