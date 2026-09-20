//! Elias-Fano encoding of a strictly increasing `u32` list.
//!
//! Each value splits into `l = floor(log2(universe / n))` low bits and a
//! high part `value >> l`. The low parts are stored contiguously; the high
//! parts are unary in a bitvector with one bit per element at position
//! `(value >> l) + index`, so the high part of element `i` equals the number
//! of zeros before its one.
//!
//! Long bitvectors carry two-level select-one samples (`u32` every 1024 ones,
//! `u16` every 64) for `get`, and matching select-zero samples for seek.
//!
//! Byte layout, all sections byte-aligned:
//!
//! ```text
//! [l: u8][lows: n*l bits][highs: n + (universe >> l) bits]
//! [select1 super: ceil(n / SELECT1_SUPER) * u32][select1 sub: ceil(n / SELECT1_SUB) * u16]
//! [select0 super: ceil(z / SELECT0_SUPER) * u32]
//! [select0 sub: ceil(z / SELECT0_SUB) * u16]   z = universe >> l
//! ```
//!
//! Samples are omitted when the highs fit in 16 words: scanning costs less
//! than carrying an index across short lists.
//!
//! `n` and `universe` are stored out of band by the caller. Empty lists
//! encode to a single `l = 0` byte and decode to nothing.

use crate::bits::{self, Reader, Writer};

mod view;
pub use view::View;

pub(crate) const SELECT1_SUPER: usize = 1024;
const SELECT1_SUB: usize = 64;
pub(crate) const SAMPLE_THRESHOLD_WORDS: usize = 16;

pub(crate) const SELECT0_SUPER: usize = 1024;
pub(crate) const SELECT0_SUB: usize = 64;

pub(crate) fn has_samples(highs_bits: usize) -> bool {
    highs_bits.div_ceil(64) > SAMPLE_THRESHOLD_WORDS
}

pub(crate) fn select1_len(n: usize, highs_bits: usize) -> usize {
    usize::from(has_samples(highs_bits))
        * (n.div_ceil(SELECT1_SUPER) * 4 + n.div_ceil(SELECT1_SUB) * 2)
}

/// `floor(log2(universe / n))`, the low-bit width that caps the high
/// bitvector at between `2n` and `3n` bits.
pub(crate) fn low_width(universe: u32, n: usize) -> u32 {
    if n == 0 {
        return 0;
    }
    let quotient = u64::from(universe) / n as u64;
    (u64::BITS - quotient.leading_zeros()).saturating_sub(1)
}

/// Bare byte-aligned `[lows][highs]`, shared with PEF partitions. `base`
/// shifts partition-local values.
pub(crate) fn encode_parts(universe: u32, l: u32, base: u32, list: &[u32], out: &mut Vec<u8>) {
    let n = list.len();
    let mut writer = Writer::new(out);
    let low_mask = (1u64 << l) - 1;
    for &v in list {
        writer.put(u64::from(v.wrapping_sub(base)) & low_mask, l);
    }
    writer.align();
    let highs_bits = n + (u64::from(universe) >> l) as usize;
    let highs_start = out.len();
    out.resize(highs_start + highs_bits.div_ceil(8), 0);
    for (i, &v) in list.iter().enumerate() {
        let pos = (u64::from(v.wrapping_sub(base)) >> l) as usize + i;
        out[highs_start + pos / 8] |= 1 << (pos % 8);
    }
}

/// Decode a bare `[lows][highs]` pair produced by [`encode_parts`], adding
/// `base` back.
pub(crate) fn decode_parts(l: u32, n: usize, buf: &[u8], base: u32, out: &mut Vec<u32>) {
    if n == 0 {
        return;
    }
    let lows_bytes = (n * l as usize).div_ceil(8);
    let highs_bit = lows_bytes * 8;
    let reader = Reader::new(buf);
    let mut index = 0usize;
    let mut bit = highs_bit;
    while index < n {
        let mut word = reader.word(bit);
        while word != 0 && index < n {
            let one_pos = bit + word.trailing_zeros() as usize;
            let high = (one_pos - highs_bit - index) as u32;
            let low = reader.get(index * l as usize, l) as u32;
            out.push((((u64::from(high) << l) | u64::from(low)) as u32).wrapping_add(base));
            index += 1;
            word &= word - 1;
        }
        bit += 64;
    }
}

/// Append the Elias-Fano encoding of `list` (values in `[0, universe)`) to
/// `out`: the `l` byte, the lows and highs, then the select samples.
pub fn encode(universe: u32, list: &[u32], out: &mut Vec<u8>) {
    let n = list.len();
    let l = low_width(universe, n);
    let list_start = out.len();
    out.push(l as u8);
    if n == 0 {
        return;
    }
    let lows_bytes = (n * l as usize).div_ceil(8);
    let highs_start = 1 + lows_bytes;
    let highs_bits = n + (u64::from(universe) >> l) as usize;
    encode_parts(universe, l, 0, list, out);
    if has_samples(highs_bits) {
        encode_select1(list, l, out);
        let highs_slice = &out[list_start + highs_start..];
        out.extend_from_slice(&encode_select0_bytes(universe, l, n, highs_slice));
    }
}

pub(crate) fn encode_select1(list: &[u32], l: u32, out: &mut Vec<u8>) {
    for (i, &value) in list.iter().enumerate().step_by(SELECT1_SUPER) {
        let pos = (u64::from(value) >> l) as usize + i;
        out.extend_from_slice(&(pos as u32).to_le_bytes());
    }
    for (i, &value) in list.iter().enumerate().step_by(SELECT1_SUB) {
        let super_rank = i / SELECT1_SUPER * SELECT1_SUPER;
        let base = (u64::from(list[super_rank]) >> l) as usize + super_rank;
        let pos = (u64::from(value) >> l) as usize + i;
        let offset = u16::try_from(pos - base).unwrap_or(u16::MAX);
        out.extend_from_slice(&offset.to_le_bytes());
    }
}

/// Select-zero super/sub samples, built by walking the highs' zero bits.
pub(crate) fn encode_select0_bytes(universe: u32, l: u32, n: usize, highs: &[u8]) -> Vec<u8> {
    let zeros = (u64::from(universe) >> l) as usize;
    let highs_bits = n + zeros;
    let mut super_samples = Vec::with_capacity(zeros.div_ceil(SELECT0_SUPER));
    let mut sub_samples = Vec::with_capacity(zeros.div_ceil(SELECT0_SUB));
    let mut zeros_seen = 0usize;
    let mut super_base = 0usize;
    for word_index in 0..highs_bits.div_ceil(64) {
        let mut word = !word_at(highs, word_index * 64);
        if word_index * 64 + 64 > highs_bits {
            word &= trailing_mask(highs_bits - word_index * 64);
        }
        while word != 0 {
            let pos = word_index * 64 + word.trailing_zeros() as usize;
            if zeros_seen.is_multiple_of(SELECT0_SUPER) {
                super_base = pos;
                super_samples.push(pos as u32);
            }
            if zeros_seen.is_multiple_of(SELECT0_SUB) {
                sub_samples.push((pos - super_base) as u16);
            }
            zeros_seen += 1;
            word &= word - 1;
        }
    }
    debug_assert_eq!(zeros_seen, zeros);
    let mut out = Vec::with_capacity(super_samples.len() * 4 + sub_samples.len() * 2);
    for &s in &super_samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    for &s in &sub_samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Reconstruct exactly the `n` values `encode` produced for them, appending
/// to `out`.
pub fn decode(universe: u32, n: usize, buf: &[u8], out: &mut Vec<u32>) {
    let _ = universe;
    if n == 0 {
        return;
    }
    let l = u32::from(buf[0]);
    decode_parts(l, n, &buf[1..], 0, out);
}
/// Bytes the select samples occupy, reported as the index overhead: the
/// select-one samples plus the select-zero super/sub samples.
pub fn aux_len(universe: u32, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let l = low_width(universe, n);
    let zeros = (u64::from(universe) >> l) as usize;
    let highs_bits = n + zeros;
    if has_samples(highs_bits) {
        select1_len(n, highs_bits)
            + zeros.div_ceil(SELECT0_SUPER) * 4
            + zeros.div_ceil(SELECT0_SUB) * 2
    } else {
        0
    }
}
/// The `i`-th value by ordinal, via the select samples.
pub fn get(universe: u32, n: usize, buf: &[u8], i: usize) -> u32 {
    View::new(universe, n, buf).get(i)
}

/// Byte offsets of the select sample structures within a buffer.
#[derive(Default)]
pub(crate) struct SampleIndex {
    pub(crate) select1_super: usize,
    pub(crate) select1_sub: usize,
    pub(crate) select0_super: usize,
    pub(crate) select0_sub: usize,
}

/// Parsed byte offsets for a full EF list or a bare PEF `[lows][highs]`.
#[derive(Clone, Copy)]
pub(crate) struct Layout<'a> {
    pub(crate) buf: &'a [u8],
    pub(crate) n: usize,
    pub(crate) l: u32,
    lows_bit: usize,
    highs_bit: usize,
    highs_bits: usize,
    s1_super_start: usize,
    s1_sub_start: usize,
    sample_count: usize,
    s0_super_start: usize,
    s0_sub_start: usize,
}

impl<'a> Layout<'a> {
    /// Parse the full `[l: u8][lows][highs][select1][select0]` format.
    pub(crate) fn with_samples(universe: u32, n: usize, buf: &'a [u8]) -> Self {
        let l = u32::from(buf[0]);
        let lows_bytes = (n * l as usize).div_ceil(8);
        let highs_bits = n + (u64::from(universe) >> l) as usize;
        let samples_start = 1 + lows_bytes + highs_bits.div_ceil(8);
        let zeros = (u64::from(universe) >> l) as usize;
        let samples = if has_samples(highs_bits) {
            let select1_sub = samples_start + n.div_ceil(SELECT1_SUPER) * 4;
            let select0_super = select1_sub + n.div_ceil(SELECT1_SUB) * 2;
            SampleIndex {
                select1_super: samples_start,
                select1_sub,
                select0_super,
                select0_sub: select0_super + zeros.div_ceil(SELECT0_SUPER) * 4,
            }
        } else {
            SampleIndex::default()
        };
        Self::from_parts_at(l, n, universe, buf, 8, samples)
    }
    /// Parse a bare `[lows][highs]` pair (no `l` byte, no samples).
    pub(crate) fn from_parts(l: u32, n: usize, universe: u32, buf: &'a [u8]) -> Self {
        Self::from_parts_at(l, n, universe, buf, 0, SampleIndex::default())
    }

    /// Parse a `[lows][highs]` pair at bit `lows_bit` of `buf`, with the
    /// select samples at the byte offsets in `samples`. `pef` uses this for
    /// its upper-level EF, whose samples live after the payload.
    pub(crate) fn from_parts_at(
        l: u32,
        n: usize,
        universe: u32,
        buf: &'a [u8],
        lows_bit: usize,
        samples: SampleIndex,
    ) -> Self {
        let lows_bytes = (n * l as usize).div_ceil(8);
        let highs_bit = lows_bit / 8 * 8 + lows_bytes * 8;
        let highs_bits = n + (u64::from(universe) >> l) as usize;
        let sample_count = usize::from(samples.select1_super > 0) * n.div_ceil(SELECT1_SUPER);
        Self {
            buf,
            n,
            l,
            lows_bit,
            highs_bit,
            highs_bits,
            s1_super_start: samples.select1_super,
            s1_sub_start: samples.select1_sub,
            sample_count,
            s0_super_start: samples.select0_super,
            s0_sub_start: samples.select0_sub,
        }
    }

    /// Position of the one for the `i`-th element (its rank), jumping via the
    /// samples when present and scanning at most [`SELECT1_SUB`] ones.
    pub(crate) fn select_one(&self, i: usize) -> usize {
        select_one(
            self.buf,
            self.highs_bit,
            self.s1_super_start,
            self.s1_sub_start,
            i,
        )
    }

    /// Value at rank `i`, whose one is at `pos` relative to the highs.
    pub(crate) fn value_at(&self, i: usize, pos: usize) -> u32 {
        let high = (pos - i) as u32;
        let low = Reader::new(self.buf).get(self.lows_bit + i * self.l as usize, self.l) as u32;
        ((u64::from(high) << self.l) | u64::from(low)) as u32
    }

    /// Position of the next one strictly after `pos`, or `None` past the end.
    pub(crate) fn next_one_after(&self, pos: usize) -> Option<usize> {
        if pos >= self.highs_bits {
            return None;
        }
        let mut word_index = pos / 64;
        let mut word = self.word_at_rel(word_index) & mask_above(pos % 64 + 1);
        loop {
            if word != 0 {
                return Some(word_index * 64 + word.trailing_zeros() as usize);
            }
            word_index += 1;
            if word_index * 64 >= self.highs_bits {
                return None;
            }
            word = self.word_at_rel(word_index);
        }
    }

    /// Scan forward from rank `index` (the one at bit `pos`) for the first
    /// element with value `>= target`, returning `(index, pos, value)`.
    /// A whole 64-bit word whose last element still has a high part below the
    /// target's is skipped in one popcount; only the word containing the
    /// answer is walked element by element.
    pub(crate) fn scan(
        &self,
        mut index: usize,
        pos: usize,
        target: u32,
    ) -> Option<(usize, usize, u32)> {
        let target_high = (u64::from(target) >> self.l) as usize;
        let target_low = (u64::from(target) & ((1u64 << self.l) - 1)) as u32;
        let mut word_index = pos / 64;
        let mut word = self.word_at_rel(word_index) & mask_above(pos % 64);
        loop {
            if word == 0 {
                word_index += 1;
                word = self.word_at_rel(word_index);
                continue;
            }
            let ones = word.count_ones() as usize;
            let last_pos = word_index * 64 + (63 - word.leading_zeros()) as usize;
            if last_pos - (index + ones - 1) < target_high {
                index += ones;
                if index >= self.n {
                    return None;
                }
                word_index += 1;
                word = self.word_at_rel(word_index);
                continue;
            }
            loop {
                let one_pos = word_index * 64 + word.trailing_zeros() as usize;
                let high = one_pos - index;
                if high > target_high {
                    return Some((index, one_pos, self.value_at(index, one_pos)));
                }
                if high == target_high {
                    let low = Reader::new(self.buf)
                        .get(self.lows_bit + index * self.l as usize, self.l)
                        as u32;
                    if low >= target_low {
                        let value = ((u64::from(high as u32) << self.l) | u64::from(low)) as u32;
                        return Some((index, one_pos, value));
                    }
                }
                index += 1;
                if index >= self.n {
                    return None;
                }
                word &= word - 1;
                if word == 0 {
                    break;
                }
            }
            word_index += 1;
            word = self.word_at_rel(word_index);
        }
    }

    /// Land near `target`; PEF without select-zero binary-searches select-one.
    pub(crate) fn land(&self, target: u32) -> Option<(usize, usize)> {
        let target_high = (u64::from(target) >> self.l) as usize;
        if self.s0_super_start > 0 {
            if target_high == 0 {
                return Some((0, self.select_one(0)));
            }
            let zeros_total = self.highs_bits - self.n;
            if target_high > zeros_total {
                return None;
            }
            // The (target_high-1)-th zero (0-indexed) sits just before the
            // elements with high == target_high; the first one after it is
            // the first element with high >= target_high.
            let z = self.select_zero(target_high - 1);
            let index = z - (target_high - 1);
            let pos = self.next_one_after(z)?;
            Some((index, pos))
        } else {
            let (mut lo, mut hi) = (0usize, self.sample_count);
            while lo < hi {
                let mid = (lo + hi) / 2;
                let pos = self.super_position(mid);
                if pos - mid * SELECT1_SUPER < target_high {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            if lo == 0 {
                Some((0, self.select_one(0)))
            } else {
                Some(((lo - 1) * SELECT1_SUPER, self.super_position(lo - 1)))
            }
        }
    }

    /// Position of the `k`-th zero (0-indexed, so `k` zeros lie before it),
    /// via the super/sub samples and at most [`SELECT0_SUB`] zeros of scan.
    fn select_zero(&self, k: usize) -> usize {
        let super_index = k / SELECT0_SUPER;
        let base = u32::from_le_bytes([
            self.buf[self.s0_super_start + super_index * 4],
            self.buf[self.s0_super_start + super_index * 4 + 1],
            self.buf[self.s0_super_start + super_index * 4 + 2],
            self.buf[self.s0_super_start + super_index * 4 + 3],
        ]) as usize;
        let sub_index = k / SELECT0_SUB;
        let sub_bytes =
            &self.buf[self.s0_sub_start + sub_index * 2..self.s0_sub_start + sub_index * 2 + 2];
        let start = base + usize::from(u16::from_le_bytes([sub_bytes[0], sub_bytes[1]]));
        let mut reminder = k % SELECT0_SUB;
        let mut word_index = start / 64;
        let mut word = !self.word_at_rel(word_index) & mask_above(start % 64);
        if word_index * 64 + 64 > self.highs_bits {
            word &= trailing_mask(self.highs_bits - word_index * 64);
        }
        loop {
            let zeros = word.count_ones() as usize;
            if reminder < zeros {
                return word_index * 64 + bits::select_one(word, reminder) as usize;
            }
            reminder -= zeros;
            word_index += 1;
            word = !self.word_at_rel(word_index);
            if word_index * 64 + 64 > self.highs_bits {
                word &= trailing_mask(self.highs_bits - word_index * 64);
            }
        }
    }

    fn super_position(&self, k: usize) -> usize {
        read_u32(self.buf, self.s1_super_start + k * 4) as usize
    }

    /// The highs word `word_index` (positions relative to the highs start),
    /// masked to the `highs_bits` so padding bits past the end read as zero.
    fn word_at_rel(&self, word_index: usize) -> u64 {
        let mut word = word_at(self.buf, self.highs_bit + word_index * 64);
        if word_index * 64 + 64 > self.highs_bits {
            word &= trailing_mask(self.highs_bits - word_index * 64);
        }
        word
    }
}

fn select_one(
    buf: &[u8],
    highs_bit: usize,
    super_start: usize,
    sub_start: usize,
    i: usize,
) -> usize {
    let (start, mut rest) = if super_start > 0 {
        let super_index = i / SELECT1_SUPER;
        let base = read_u32(buf, super_start + super_index * 4) as usize;
        let b = sub_start + i / SELECT1_SUB * 2;
        let offset = usize::from(u16::from_le_bytes([buf[b], buf[b + 1]]));
        if offset == usize::from(u16::MAX) {
            (base, i - super_index * SELECT1_SUPER + 1)
        } else {
            (base + offset, i % SELECT1_SUB + 1)
        }
    } else {
        (0, i + 1)
    };
    let mut word_index = start / 64;
    let mut word = word_at(buf, highs_bit + word_index * 64) & mask_above(start % 64);
    loop {
        let ones = word.count_ones() as usize;
        if rest <= ones {
            return word_index * 64 + bits::select_one(word, rest - 1) as usize;
        }
        rest -= ones;
        word_index += 1;
        word = word_at(buf, highs_bit + word_index * 64);
    }
}

fn read_u32(buf: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]])
}

/// The 64-bit word starting at bit `pos`, zero-padded past the buffer end.
fn word_at(buf: &[u8], pos: usize) -> u64 {
    Reader::new(buf).word(pos)
}

/// Bits at or above `bit`, or zero when `bit >= 64` (the shift would
/// overflow in debug builds).
fn mask_above(bit: usize) -> u64 {
    if bit >= 64 { 0 } else { u64::MAX << bit }
}

/// The low `bits` bits set, or all ones when `bits >= 64`.
fn trailing_mask(bits: usize) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// Skip cursor over one list, with Lucene `advance` semantics.
pub struct Cursor<'a> {
    layout: Layout<'a>,
    /// Current rank; `usize::MAX` before the first element, `n` when
    /// exhausted.
    index: usize,
    pos: usize,
    cur: Option<u32>,
}

impl<'a> Cursor<'a> {
    pub fn new(universe: u32, n: usize, buf: &'a [u8]) -> Self {
        Self::from_layout(Layout::with_samples(universe, n, buf))
    }

    fn from_layout(layout: Layout<'a>) -> Self {
        Self {
            layout,
            index: usize::MAX,
            pos: 0,
            cur: None,
        }
    }

    fn exhaust(&mut self) -> Option<u32> {
        self.index = self.layout.n;
        self.cur = None;
        None
    }
}

impl crate::Cursor for Cursor<'_> {
    fn next_geq(&mut self, target: u32) -> Option<u32> {
        if let Some(c) = self.cur
            && c >= target
        {
            return Some(c);
        }
        if self.layout.n == 0 || self.index == self.layout.n {
            return None;
        }
        // Fresh cursor: jump straight to the target region via the samples.
        // Otherwise scan within the current 64-bit word when the answer is
        // there, and only jump when the target's high is further ahead.
        let (index, pos) = if self.index == usize::MAX {
            self.layout.land(target)?
        } else {
            let word_index = self.pos / 64;
            let word = self.layout.word_at_rel(word_index);
            let after = if self.pos % 64 == 63 {
                0
            } else {
                (word >> (self.pos % 64 + 1)).count_ones() as usize
            };
            let last_rank = self.index + after;
            let last_pos = word_index * 64 + (63 - word.leading_zeros()) as usize;
            if self.layout.value_at(last_rank, last_pos) >= target {
                (self.index, self.pos)
            } else {
                self.layout.land(target)?
            }
        };
        let Some((index, pos, value)) = self.layout.scan(index, pos, target) else {
            return self.exhaust();
        };
        self.index = index;
        self.pos = pos;
        self.cur = Some(value);
        Some(value)
    }

    fn next(&mut self) -> Option<u32> {
        if self.layout.n == 0 || self.index == self.layout.n {
            return None;
        }
        let (index, pos) = if self.index == usize::MAX {
            (0, self.layout.select_one(0))
        } else {
            let index = self.index + 1;
            if index >= self.layout.n {
                return self.exhaust();
            }
            let Some(pos) = self.layout.next_one_after(self.pos) else {
                return self.exhaust();
            };
            (index, pos)
        };
        let value = self.layout.value_at(index, pos);
        self.index = index;
        self.pos = pos;
        self.cur = Some(value);
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::cursor::Cursor as _;

    use super::*;

    fn roundtrip(universe: u32, list: &[u32]) {
        let mut buf = Vec::new();
        encode(universe, list, &mut buf);
        let mut back = Vec::new();
        decode(universe, list.len(), &buf, &mut back);
        assert_eq!(back, list);
        for (i, &v) in list.iter().enumerate() {
            assert_eq!(get(universe, list.len(), &buf, i), v, "get({i})");
        }
    }

    #[test]
    fn lengths_roundtrip() {
        let universe = 1 << 20;
        for n in [0, 1, 2, 64, 65, 128, 129, 1000] {
            let list: Vec<u32> = (0..n as u32).map(|i| i * 7 + 3).collect();
            roundtrip(universe, &list);
            if n > 0 && n <= 300 {
                assert_eq!(aux_len(universe, n), 0);
            }
        }
    }

    #[test]
    fn edge_lists() {
        let universe = 1 << 20;
        roundtrip(universe, &[0]);
        roundtrip(universe, &[universe - 1]);
        roundtrip(universe, &[0, universe - 1]);
    }

    #[test]
    fn dense() {
        let universe = 1 << 12;
        let list: Vec<u32> = (0..universe).collect();
        roundtrip(universe, &list);
    }

    #[test]
    fn max_universe() {
        let universe = u32::MAX - 1;
        roundtrip(universe, &[0]);
        roundtrip(universe, &[universe - 1]);
        roundtrip(universe, &[0, universe - 1]);
        let list: Vec<u32> = (0..1000).map(|i| i * 1_000_000).collect();
        roundtrip(universe, &list);
    }

    #[test]
    fn random() {
        let universe = 1 << 20;
        let mut values = BTreeSet::new();
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        while values.len() < 1000 {
            state ^= state << 7;
            state ^= state >> 9;
            state ^= state << 8;
            values.insert((state % u64::from(universe)) as u32);
        }
        let list: Vec<u32> = values.into_iter().collect();
        roundtrip(universe, &list);
    }

    #[test]
    fn get_crosses_samples() {
        let universe = u32::MAX;
        let mut list: Vec<u32> = (0..50_000).collect();
        list.extend((1 << 31)..(1 << 31) + 50_000);
        let n = list.len();
        let mut buf = Vec::new();
        encode(universe, &list, &mut buf);
        for i in [0, 63, 1023, 49_999, 50_000, 50_063, n - 1] {
            assert_eq!(get(universe, n, &buf, i), list[i]);
        }
    }

    /// The bench's `conformance` semantics on one cursor.
    fn cursor_conformance(universe: u32, list: &[u32]) {
        let mut buf = Vec::new();
        encode(universe, list, &mut buf);
        let mut cur = Cursor::new(universe, list.len(), &buf);
        for (i, &v) in list.iter().enumerate() {
            let target = if i > 0 && list[i - 1] + 1 < v {
                list[i - 1] + 1
            } else {
                v
            };
            assert_eq!(cur.next_geq(target), Some(v), "next_geq({target}) != {v}");
            assert_eq!(cur.next_geq(v), Some(v), "re-seek moved off {v}");
        }
        assert_eq!(
            cur.next_geq(universe),
            None,
            "next_geq(universe) should exhaust"
        );
        // Sequential next() from a fresh cursor walks the whole list.
        let mut cur = Cursor::new(universe, list.len(), &buf);
        for &v in list {
            assert_eq!(cur.next(), Some(v));
        }
        assert_eq!(cur.next(), None);
    }

    #[test]
    fn cursor_semantics() {
        let universe = 1 << 20;
        let lists: Vec<Vec<u32>> = vec![
            vec![],
            vec![0],
            vec![universe - 1],
            (0..300).collect(),
            (0..300).map(|i| i * 1000).collect(),
            (0..129).map(|i| i * 3 + 1).collect(),
            (0..1000).map(|i| i * i).collect(),
            vec![0, universe - 1],
            (0..3 * SELECT1_SUPER as u32).map(|i| i * 5 + 1).collect(),
        ];
        for list in lists {
            cursor_conformance(universe, &list);
        }
    }

    #[test]
    fn seeks_agree_with_reference() {
        // next_geq from a fresh cursor must equal binary search for random
        // targets, including targets in empty regions of the universe.
        let universe = 1 << 20;
        let n = 5000;
        let mut values = BTreeSet::new();
        let mut state = 0xd1b5_4a32_d192_ed03u64;
        while values.len() < n {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            values.insert((state % u64::from(universe)) as u32);
        }
        let list: Vec<u32> = values.into_iter().collect();
        let mut buf = Vec::new();
        encode(universe, &list, &mut buf);
        let mut state = 0x3c6e_f372_fe94_f82au64;
        for _ in 0..2000 {
            state ^= state << 7;
            state ^= state >> 9;
            state ^= state << 8;
            let target = (state % u64::from(universe)) as u32;
            let mut cur = Cursor::new(universe, n, &buf);
            let got = cur.next_geq(target);
            let expect = list.iter().copied().find(|&v| v >= target);
            assert_eq!(got, expect, "seek({target})");
        }
    }

    #[test]
    fn sparse_highs_land_without_full_walk() {
        // Two dense clusters far apart: a seek into the gap must land via a
        // sample and not scan the whole high bitvector.
        let universe: u32 = 1 << 30;
        let mut list: Vec<u32> = (0..SELECT1_SUPER as u32 + 10).collect();
        list.extend(universe / 2..universe / 2 + SELECT1_SUPER as u32);
        let mut buf = Vec::new();
        encode(universe, &list, &mut buf);
        let mut cur = Cursor::new(universe, list.len(), &buf);
        assert_eq!(cur.next_geq(universe / 4), Some(universe / 2));
    }
}
