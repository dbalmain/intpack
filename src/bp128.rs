//! Bitpacked 128-blocks with a varint tail — the Lucene/Tantivy posting-list
//! shape, and the container over the [`bitpack`] primitive.
//!
//! Layout: every full block of 128 values is `[width: u8]` then
//! `bitpack::packed_len(width)` packed bytes; the `n % 128` values after the
//! last full block are LEB128 varints. Sorted lists store gaps per [`delta`]:
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
use crate::{Cursor, delta};

// ── blocks ──

/// Encode one full block of a sorted list, gaps relative to `base` (the
/// previous block's last value, `None` for the first block). Returns the
/// block's last value, the next block's base.
pub(crate) fn encode_block_sorted(
    base: Option<u32>,
    block: &[u32; BLOCK],
    out: &mut Vec<u8>,
) -> u32 {
    let width = bitpack::width_sorted(base, block);
    let packed = reserve_block(width, out);
    bitpack::pack_sorted(width, base, block, packed);
    block[BLOCK - 1]
}

/// Encode one full block of values as-is.
pub(crate) fn encode_block(block: &[u32; BLOCK], out: &mut Vec<u8>) {
    let width = bitpack::width(block);
    bitpack::pack(width, block, reserve_block(width, out));
}

/// Append the width byte and room for the packed bytes, returning the room.
fn reserve_block(width: u32, out: &mut Vec<u8>) -> &mut [u8] {
    out.push(width as u8);
    let start = out.len();
    out.resize(start + bitpack::packed_len(width), 0);
    &mut out[start..]
}

/// Decode the full block whose width byte is at `buf[*byte]`, advancing
/// `*byte` past it. Returns the block's last value.
pub(crate) fn decode_block_sorted(
    base: Option<u32>,
    buf: &[u8],
    byte: &mut usize,
    out: &mut [u32; BLOCK],
) -> u32 {
    let (width, packed) = take_block(buf, byte);
    bitpack::unpack_sorted(width, base, packed, out);
    out[BLOCK - 1]
}

/// Unsorted [`decode_block_sorted`].
pub(crate) fn decode_block(buf: &[u8], byte: &mut usize, out: &mut [u32; BLOCK]) {
    let (width, packed) = take_block(buf, byte);
    bitpack::unpack(width, packed, out);
}

/// The width and packed bytes of the block at `buf[*byte]`, advancing
/// `*byte` past it.
fn take_block<'a>(buf: &'a [u8], byte: &mut usize) -> (u32, &'a [u8]) {
    let width = u32::from(buf[*byte]);
    let packed = &buf[*byte + 1..*byte + 1 + bitpack::packed_len(width)];
    *byte += 1 + packed.len();
    (width, packed)
}

/// Bytes the block at `buf[byte]` occupies, width byte included.
pub(crate) fn block_len(buf: &[u8], byte: usize) -> usize {
    1 + bitpack::packed_len(u32::from(buf[byte]))
}

// ── tail ──

/// LEB128 varints for the partial block: 7 bits per byte, low bits first,
/// high bit set on every byte but the last.
pub(crate) mod varint {
    pub fn put(mut v: u32, out: &mut Vec<u8>) {
        while v >= 0x80 {
            out.push(v as u8 | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }

    /// Read the varint at `buf[*pos]`, advancing `*pos` past it.
    pub fn take(buf: &[u8], pos: &mut usize) -> u32 {
        let mut v = 0;
        let mut shift = 0;
        loop {
            let b = buf[*pos];
            *pos += 1;
            v |= u32::from(b & 0x7f) << shift;
            if b < 0x80 {
                return v;
            }
            shift += 7;
        }
    }
}

/// Encode the sorted tail as gaps from `base`, the last full block's last
/// value (`None` when the list has no full block).
pub(crate) fn encode_tail_sorted(base: Option<u32>, tail: &[u32], out: &mut Vec<u8>) {
    let mut prev = base;
    for &v in tail {
        varint::put(delta::gap(prev, v), out);
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
        let v = delta::value(prev, varint::take(buf, byte));
        out.push(v);
        prev = Some(v);
    }
}

pub(crate) fn encode_tail(tail: &[u32], out: &mut Vec<u8>) {
    for &v in tail {
        varint::put(v, out);
    }
}

pub(crate) fn decode_tail(buf: &[u8], byte: &mut usize, n: usize, out: &mut Vec<u32>) {
    for _ in 0..n {
        out.push(varint::take(buf, byte));
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
            delta::value(self.cur, varint::take(self.buf, &mut self.byte))
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
