//! [`pfor128`] with a skip table in front: one `(last_value, byte_offset)`
//! entry per full block lets a cursor binary-search for the block holding a
//! target and unpack only that one, and lets `get` unpack one block by
//! ordinal. The payload after the table is byte-identical to `pfor128`.
//!
//! Layout, little-endian: `n / 128` entries of `last_value: u32` (the block's
//! final element, which is the next block's delta base; `0` for unsorted
//! lists) then `byte_offset: u32` (the block's token byte, relative to the
//! payload start); then the `pfor128` payload. The table's length follows from
//! `n`, so a list shorter than one block is byte-identical to `pfor128`.
//!
//! [`pfor128`]: crate::pfor128

use crate::bitpack::BLOCK;
use crate::pfor128;
use crate::{Cursor, delta, vbyte};

/// Bytes of skip table in front of the payload of an `n`-element list.
pub fn aux_len(n: usize) -> usize {
    n / BLOCK * 8
}

/// The skip table of an `n`-element list.
#[derive(Clone, Copy)]
struct Table<'a> {
    buf: &'a [u8],
    /// The table's `u32`s: entry `b` is `words[2 * b]` and `words[2 * b + 1]`.
    words: &'a [[u8; 4]],
    blocks: usize,
}

impl<'a> Table<'a> {
    fn new(n: usize, buf: &'a [u8]) -> Self {
        Self {
            buf,
            words: buf[..aux_len(n)].as_chunks::<4>().0,
            blocks: n / BLOCK,
        }
    }

    /// Block `b`'s last value.
    fn last(&self, b: usize) -> u32 {
        u32::from_le_bytes(self.words[2 * b])
    }

    /// Where block `b` starts in `buf`.
    fn block_start(&self, b: usize) -> usize {
        self.blocks * 8 + u32::from_le_bytes(self.words[2 * b + 1]) as usize
    }

    /// Delta base of block `b`: the previous block's last value.
    fn base(&self, b: usize) -> Option<u32> {
        (b > 0).then(|| self.last(b - 1))
    }

    /// Where the tail starts in `buf`: just past the last block.
    fn tail_start(&self) -> usize {
        match self.blocks.checked_sub(1) {
            Some(b) => self.block_start(b) + pfor128::block_len(self.buf, self.block_start(b)),
            None => 0,
        }
    }

    /// First block at or after `from` whose last value is `>= target`, or
    /// `blocks` if none: the tail, if anything, holds the target.
    fn find(&self, from: usize, target: u32) -> usize {
        let (mut lo, mut hi) = (from, self.blocks);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.last(mid) < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

/// Reserve the table and encode every full block through `block`, then the
/// rest through `tail`; both get the previous block's last value (which
/// `block` returns) as the delta base, `None` for the first block.
fn encode_with(
    list: &[u32],
    out: &mut Vec<u8>,
    block: impl Fn(Option<u32>, &[u32; BLOCK], &mut Vec<u8>) -> u32,
    tail: impl FnOnce(Option<u32>, &[u32], &mut Vec<u8>),
) {
    let (blocks, rest) = list.as_chunks::<BLOCK>();
    let table = out.len();
    out.resize(table + aux_len(list.len()), 0);
    let payload = out.len();
    let mut base = None;
    for (b, values) in blocks.iter().enumerate() {
        let offset = (out.len() - payload) as u32;
        let last = block(base, values, out);
        let entry = table + b * 8;
        out[entry..entry + 4].copy_from_slice(&last.to_le_bytes());
        out[entry + 4..entry + 8].copy_from_slice(&offset.to_le_bytes());
        base = Some(last);
    }
    tail(base, rest, out);
}

/// Encode a strictly increasing list, appending to `out`.
pub fn encode_sorted(list: &[u32], out: &mut Vec<u8>) {
    encode_with(
        list,
        out,
        pfor128::encode_block_sorted,
        pfor128::encode_tail_sorted,
    );
}

/// Decode the `n` values of a sorted list from `buf`, appending to `out`.
pub fn decode_sorted(n: usize, buf: &[u8], out: &mut Vec<u32>) {
    pfor128::decode_sorted(n, &buf[aux_len(n)..], out);
}

/// Encode an arbitrary list, appending to `out`; table entries store `0` as
/// the last value.
pub fn encode(list: &[u32], out: &mut Vec<u8>) {
    encode_with(
        list,
        out,
        |_, block, out| {
            pfor128::encode_block(block, out);
            0
        },
        |_, tail, out| pfor128::encode_tail(tail, out),
    );
}

/// Decode the `n` values of an unsorted list from `buf`, appending to `out`.
pub fn decode(n: usize, buf: &[u8], out: &mut Vec<u32>) {
    pfor128::decode(n, &buf[aux_len(n)..], out);
}

/// Element `i` of the sorted list: unpacks one block, or walks the tail.
/// Panics if `i >= n`.
pub fn get_sorted(n: usize, buf: &[u8], i: usize) -> u32 {
    assert!(i < n, "get({i}) on a list of {n}");
    let table = Table::new(n, buf);
    let b = i / BLOCK;
    if b < table.blocks {
        let mut block = [0; BLOCK];
        pfor128::decode_block_sorted(table.base(b), buf, &mut table.block_start(b), &mut block);
        return block[i % BLOCK];
    }
    let mut byte = table.tail_start();
    let mut prev = delta::carried(table.base(table.blocks));
    for _ in 0..=i % BLOCK {
        prev = delta::value_after(prev, vbyte::take(buf, &mut byte));
    }
    prev
}

/// Element `i` of the unsorted list. Panics if `i >= n`.
pub fn get(n: usize, buf: &[u8], i: usize) -> u32 {
    assert!(i < n, "get({i}) on a list of {n}");
    let table = Table::new(n, buf);
    let b = i / BLOCK;
    if b < table.blocks {
        let mut block = [0; BLOCK];
        pfor128::decode_block(buf, &mut table.block_start(b), &mut block);
        return block[i % BLOCK];
    }
    let mut byte = table.tail_start();
    let mut v = 0;
    for _ in 0..=i % BLOCK {
        v = vbyte::take(buf, &mut byte);
    }
    v
}

// ── cursor ──

/// Skip cursor over a sorted list: `next_geq` binary-searches the table from
/// the current block and unpacks only the block it lands in; the tail is
/// entered from the table's last entry without unpacking anything.
pub struct SortedCursor<'a> {
    buf: &'a [u8],
    n: usize,
    table: Table<'a>,
    /// Elements consumed so far; the current element is `pos - 1`.
    pos: usize,
    block: [u32; BLOCK],
    /// Which block `block` holds, `usize::MAX` for none yet.
    loaded: usize,
    /// Where the next tail varint starts, once the tail has been entered.
    tail: Option<usize>,
    /// The current element, `None` before the first and after the last.
    cur: Option<u32>,
    #[cfg(test)]
    unpacked: usize,
}

impl<'a> SortedCursor<'a> {
    /// Open the `n`-element list encoded in `buf`, positioned before its
    /// first element.
    pub fn new(n: usize, buf: &'a [u8]) -> Self {
        Self {
            buf,
            n,
            table: Table::new(n, buf),
            pos: 0,
            block: [0; BLOCK],
            loaded: usize::MAX,
            tail: None,
            cur: None,
            #[cfg(test)]
            unpacked: 0,
        }
    }

    /// Have `block` hold block `b`. The check is here and the decode kept out
    /// of line so the per-element callers inline it and pay a call only per
    /// block.
    fn load(&mut self, b: usize) {
        if self.loaded != b {
            self.unpack(b);
        }
    }

    #[inline(never)]
    fn unpack(&mut self, b: usize) {
        pfor128::decode_block_sorted(
            self.table.base(b),
            self.buf,
            &mut self.table.block_start(b),
            &mut self.block,
        );
        self.loaded = b;
        #[cfg(test)]
        {
            self.unpacked += 1;
        }
    }

    /// Position at the end of the last full block, ready to read the tail;
    /// the delta base is the table's last entry.
    fn enter_tail(&mut self) {
        self.pos = self.table.blocks * BLOCK;
        self.cur = self.table.base(self.table.blocks);
        self.tail = Some(self.table.tail_start());
    }

    /// Scan the tail, already entered, for the first value `>= target`.
    fn scan_tail(&mut self, target: u32) -> Option<u32> {
        let mut byte = self.tail.unwrap_or_default();
        let mut prev = delta::carried(self.cur);
        let mut pos = self.pos;
        let mut hit = false;
        while pos < self.n {
            prev = delta::value_after(prev, vbyte::take(self.buf, &mut byte));
            pos += 1;
            if prev >= target {
                hit = true;
                break;
            }
        }
        self.pos = pos;
        self.tail = Some(byte);
        self.cur = hit.then_some(prev);
        self.cur
    }
}

impl Cursor for SortedCursor<'_> {
    fn next_geq(&mut self, target: u32) -> Option<u32> {
        if let Some(cur) = self.cur
            && cur >= target
        {
            return Some(cur);
        }
        if self.tail.is_some() {
            return self.scan_tail(target);
        }
        // Targets never decrease, so the search starts at the current block.
        let from = self.pos.saturating_sub(1) / BLOCK;
        let b = self.table.find(from, target);
        if b == self.table.blocks {
            self.enter_tail();
            return self.scan_tail(target);
        }
        self.load(b);
        let start = if b == from { self.pos % BLOCK } else { 0 };
        for j in start..BLOCK {
            if self.block[j] >= target {
                self.pos = b * BLOCK + j + 1;
                self.cur = Some(self.block[j]);
                return self.cur;
            }
        }
        unreachable!("block {b} ends at or past {target}")
    }

    fn next(&mut self) -> Option<u32> {
        if self.pos == self.n {
            self.cur = None;
            return None;
        }
        let v = if self.pos < self.table.blocks * BLOCK {
            self.load(self.pos / BLOCK);
            self.block[self.pos % BLOCK]
        } else {
            if self.tail.is_none() {
                self.enter_tail();
            }
            let mut byte = self.tail.unwrap_or_default();
            let v = delta::value(self.cur, vbyte::take(self.buf, &mut byte));
            self.tail = Some(byte);
            v
        };
        self.pos += 1;
        self.cur = Some(v);
        self.cur
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pfor128::tests::{check_cursor, sorted_cases, unsorted_cases};

    #[test]
    fn sorted_roundtrip() {
        for list in sorted_cases() {
            let mut buf = Vec::new();
            encode_sorted(&list, &mut buf);
            let mut back = Vec::new();
            decode_sorted(list.len(), &buf, &mut back);
            assert_eq!(back, list, "n = {}", list.len());
            for (i, &v) in list.iter().enumerate() {
                assert_eq!(get_sorted(list.len(), &buf, i), v, "get({i})");
            }
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
            for (i, &v) in list.iter().enumerate() {
                assert_eq!(get(list.len(), &buf, i), v, "get({i})");
            }
        }
    }

    #[test]
    fn table_then_pfor128_payload() {
        let list: Vec<u32> = (0..300).map(|i| i * 7).collect();
        let mut buf = Vec::new();
        encode_sorted(&list, &mut buf);
        let mut plain = Vec::new();
        pfor128::encode_sorted(&list, &mut plain);
        assert_eq!(aux_len(list.len()), 16);
        assert_eq!(buf[16..], plain);
        let table = Table::new(list.len(), &buf);
        assert_eq!((table.last(0), table.last(1)), (list[127], list[255]));
        assert_eq!(table.block_start(0), 16);
        assert_eq!(table.block_start(1), 16 + 1 + 16 * 3);
        assert_eq!(table.tail_start(), plain.len() - 44 + 16);
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

    #[test]
    fn seek_unpacks_only_the_landing_block() {
        let list: Vec<u32> = (0..12800).map(|i| i * 100).collect();
        let mut buf = Vec::new();
        encode_sorted(&list, &mut buf);
        let mut cursor = SortedCursor::new(list.len(), &buf);
        assert_eq!(cursor.next_geq(1_152_005), Some(1_152_100));
        assert_eq!(cursor.unpacked, 1);
        assert_eq!(cursor.next_geq(1_160_000), Some(1_160_000), "same block");
        assert_eq!(cursor.unpacked, 1);
        assert_eq!(cursor.next_geq(1_280_000 - 100), list.last().copied());
        assert_eq!(cursor.unpacked, 2);
    }

    #[test]
    fn patched_blocks_support_seek_and_get() {
        let mut gaps = [0u32; BLOCK];
        gaps[64] = 86_000;
        delta::decode(None, &mut gaps);
        let list = gaps.to_vec();
        let mut buf = Vec::new();
        encode_sorted(&list, &mut buf);
        let mut cursor = SortedCursor::new(list.len(), &buf);
        assert_eq!(cursor.next_geq(list[63] + 1), Some(list[64]));
        assert_eq!(get_sorted(list.len(), &buf, 64), list[64]);
        assert_eq!(get_sorted(list.len(), &buf, 127), list[127]);
    }

    #[test]
    fn raw_escapes_support_unsorted_get() {
        for mut block in [[0; BLOCK], [1 << 30; BLOCK]] {
            if block[7] == 0 {
                block[..8].fill(u32::MAX);
            } else {
                block[..7].fill(u32::MAX);
            }
            let mut buf = Vec::new();
            encode(&block, &mut buf);
            assert_eq!(buf[aux_len(BLOCK)], u8::MAX);
            let mut back = Vec::new();
            decode(BLOCK, &buf, &mut back);
            assert_eq!(back, block);
            for i in [0, 6, 7, BLOCK - 1] {
                assert_eq!(get(BLOCK, &buf, i), block[i]);
            }
        }
    }

    #[test]
    fn tail_needs_no_block() {
        let list: Vec<u32> = (0..256 + 60).map(|i| i * 7 + 1).collect();
        let mut buf = Vec::new();
        encode_sorted(&list, &mut buf);
        let mut cursor = SortedCursor::new(list.len(), &buf);
        assert_eq!(cursor.next_geq(list[256]), Some(list[256]));
        assert_eq!(cursor.unpacked, 0);
        assert_eq!(cursor.next_geq(list[300]), Some(list[300]));
        assert_eq!(cursor.next_geq(u32::MAX), None);
        assert_eq!(cursor.next(), None);
    }
}
