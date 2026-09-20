//! Elias-Fano encoding of a strictly increasing `u32` list with a sampled
//! select index, so a fresh seek jumps instead of walking the highs.
//!
//! Each value splits into `l = floor(log2(universe / n))` low bits and a
//! high part `value >> l`. The low parts are stored contiguously; the high
//! parts are unary in a bitvector with one bit per element at position
//! `(value >> l) + index`, so the high part of element `i` equals the number
//! of zeros before its one. After the bitvector, the position of every
//! [`SELECT_Q`]th one is stored as a `u32` — a select index costing 4 bytes
//! per 256 elements (~0.02 bits/int), which is what a seek needs to skip a
//! whole sparse region without walking it.
//!
//! Byte layout, all sections byte-aligned:
//!
//! ```text
//! [l: u8][lows: n*l bits][highs: n + (universe >> l) bits][select samples: ceil(n / SELECT_Q) * u32]
//! ```
//!
//! `n` and `universe` are stored out of band by the caller. Empty lists
//! encode to a single `l = 0` byte and decode to nothing.
//!
//! [`SELECT_Q`]: self::SELECT_Q

use crate::bits::{Reader, Writer};

/// Elements between select samples. A sample costs 4 bytes and the post-seek
/// scan is bounded by [`SELECT_Q`] ones, so 256 keeps the index at
/// ~0.02 bits/int while a seek lands within a few words.
pub(crate) const SELECT_Q: usize = 256;

/// `floor(log2(universe / n))`, the low-bit width that caps the high
/// bitvector at between `2n` and `3n` bits.
pub(crate) fn low_width(universe: u32, n: usize) -> u32 {
    if n == 0 {
        return 0;
    }
    let quotient = u64::from(universe) / n as u64;
    (u64::BITS - quotient.leading_zeros()).saturating_sub(1)
}

/// The shared low/high pair: `[lows][highs]` with no `l` byte and no select
/// samples, byte-aligned between the sections. This is what a [`pef`] EF
/// partition stores, so both codecs build their payloads with one routine.
/// `base` shifts the values (partition-local values are `value - start`).
///
/// [`pef`]: crate::pef
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
    out.push(l as u8);
    if n == 0 {
        return;
    }
    encode_parts(universe, l, 0, list, out);
    for (i, &v) in list.iter().enumerate().step_by(SELECT_Q) {
        let pos = (u64::from(v) >> l) as usize + i;
        out.extend_from_slice(&(pos as u32).to_le_bytes());
    }
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

/// Bytes the select samples occupy, reported as the index overhead.
pub fn aux_len(universe: u32, n: usize) -> usize {
    let _ = universe;
    n.div_ceil(SELECT_Q) * 4
}

/// The `i`-th value by ordinal, via the select samples.
pub fn get(universe: u32, n: usize, buf: &[u8], i: usize) -> u32 {
    assert!(i < n, "elias_fano::get({i}) out of range (n={n})");
    let layout = Layout::with_samples(universe, n, buf);
    let pos = layout.select_one(i);
    layout.value_at(i, pos)
}

/// Layout of one encoded list, shared between the full format and a bare
/// `[lows][highs]` sub-slice ([`Layout::from_parts`], used by [`pef`]).
///
/// [`pef`]: crate::pef
pub(crate) struct Layout<'a> {
    pub(crate) buf: &'a [u8],
    pub(crate) n: usize,
    pub(crate) l: u32,
    lows_bit: usize,
    /// Bit offset of the highs within `buf`. One positions are relative to
    /// this, so an element's high part is `pos - index`.
    highs_bit: usize,
    samples_start: usize,
    sample_count: usize,
}

impl<'a> Layout<'a> {
    /// Parse the full `[l: u8][lows][highs][samples]` format.
    pub(crate) fn with_samples(universe: u32, n: usize, buf: &'a [u8]) -> Self {
        let l = u32::from(buf[0]);
        let lows_bytes = (n * l as usize).div_ceil(8);
        let highs_bits = n + (u64::from(universe) >> l) as usize;
        let samples_start = 1 + lows_bytes + highs_bits.div_ceil(8);
        Self::from_parts_at(l, n, buf, 8, samples_start)
    }

    /// Parse a bare `[lows][highs]` pair (no `l` byte, no samples).
    pub(crate) fn from_parts(l: u32, n: usize, buf: &'a [u8]) -> Self {
        Self::from_parts_at(l, n, buf, 0, 0)
    }

    /// Parse a `[lows][highs]` pair at bit `lows_bit` of `buf`, with the
    /// select samples at byte `samples_start` (0 when absent). `pef` uses
    /// this for its upper-level EF, whose samples live after the payload.
    pub(crate) fn from_parts_at(
        l: u32,
        n: usize,
        buf: &'a [u8],
        lows_bit: usize,
        samples_start: usize,
    ) -> Self {
        let lows_bytes = (n * l as usize).div_ceil(8);
        let highs_bit = lows_bit / 8 * 8 + lows_bytes * 8;
        let sample_count = if samples_start > 0 {
            n.div_ceil(SELECT_Q)
        } else {
            0
        };
        Self {
            buf,
            n,
            l,
            lows_bit,
            highs_bit,
            samples_start,
            sample_count,
        }
    }

    /// Position of the one for the `i`-th element (its rank), jumping via the
    /// samples when present and scanning at most [`SELECT_Q`] ones otherwise.
    pub(crate) fn select_one(&self, i: usize) -> usize {
        let (start, mut rest) = if self.sample_count > 0 {
            (self.sample_position(i / SELECT_Q), i % SELECT_Q + 1)
        } else {
            (0, i + 1)
        };
        let mut word_index = start / 64;
        let mut word = self.word_at_rel(word_index) & (u64::MAX << (start % 64));
        loop {
            let ones = word.count_ones() as usize;
            if rest <= ones {
                let mut w = word;
                for _ in 0..rest - 1 {
                    w &= w - 1;
                }
                return word_index * 64 + w.trailing_zeros() as usize;
            }
            rest -= ones;
            word_index += 1;
            word = self.word_at_rel(word_index);
        }
    }

    /// The element at rank `i` whose one is at bit `pos` (relative to the
    /// highs).
    pub(crate) fn value_at(&self, i: usize, pos: usize) -> u32 {
        let high = (pos - i) as u32;
        let low = Reader::new(self.buf).get(self.lows_bit + i * self.l as usize, self.l) as u32;
        ((u64::from(high) << self.l) | u64::from(low)) as u32
    }

    /// Position of the next one strictly after `pos`.
    pub(crate) fn next_one_after(&self, pos: usize) -> usize {
        let mut word_index = pos / 64;
        let mut word = self.word_at_rel(word_index) & mask_above(pos % 64 + 1);
        loop {
            if word != 0 {
                return word_index * 64 + word.trailing_zeros() as usize;
            }
            word_index += 1;
            word = self.word_at_rel(word_index);
        }
    }

    /// Scan forward from rank `index` (the one at bit `pos`) for the first
    /// element with value `>= target`, returning `(index, pos, value)`.
    /// The scan stops at the first element whose high exceeds the target's,
    /// since that value is already past the target.
    pub(crate) fn scan(
        &self,
        mut index: usize,
        pos: usize,
        target: u32,
    ) -> Option<(usize, usize, u32)> {
        let target_high = (u64::from(target) >> self.l) as usize;
        let target_low = (u64::from(target) & ((1u64 << self.l) - 1)) as u32;
        let mut word_index = pos / 64;
        let mut word = self.word_at_rel(word_index) & (u64::MAX << (pos % 64));
        loop {
            if word != 0 {
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
                    word_index += 1;
                    word = self.word_at_rel(word_index);
                }
            } else {
                word_index += 1;
                word = self.word_at_rel(word_index);
            }
        }
    }

    /// Land just before the first element with high part `>= target >> l`,
    /// returning a rank/position to start a forward scan from. Uses the
    /// samples to binary-search the high part, so a seek never walks the
    /// highs linearly.
    pub(crate) fn land(&self, target: u32) -> Option<(usize, usize)> {
        let target_high = (u64::from(target) >> self.l) as usize;
        let (mut lo, mut hi) = (0usize, self.sample_count);
        while lo < hi {
            let mid = (lo + hi) / 2;
            let pos = self.sample_position(mid);
            if pos - mid * SELECT_Q < target_high {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo == 0 {
            return Some((0, self.sample_position(0)));
        }
        Some(((lo - 1) * SELECT_Q, self.sample_position(lo - 1)))
    }

    fn sample_position(&self, k: usize) -> usize {
        let b = self.samples_start + k * 4;
        u32::from_le_bytes([
            self.buf[b],
            self.buf[b + 1],
            self.buf[b + 2],
            self.buf[b + 3],
        ]) as usize
    }

    /// The highs word `word_index` (positions relative to the highs start).
    fn word_at_rel(&self, word_index: usize) -> u64 {
        word_at(self.buf, self.highs_bit + word_index * 64)
    }
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
        Self {
            layout: Layout::with_samples(universe, n, buf),
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
            (index, self.layout.next_one_after(self.pos))
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
        let universe = 1 << 30;
        let n = 3 * SELECT_Q + 10;
        let list: Vec<u32> = (0..n as u32).map(|i| i * 37 + 5).collect();
        let mut buf = Vec::new();
        encode(universe, &list, &mut buf);
        for (i, &v) in list.iter().enumerate() {
            assert_eq!(get(universe, n, &buf, i), v);
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
            (0..3 * SELECT_Q as u32).map(|i| i * 5 + 1).collect(),
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
        let mut list: Vec<u32> = (0..SELECT_Q as u32 + 10).collect();
        list.extend(universe / 2..universe / 2 + SELECT_Q as u32);
        let mut buf = Vec::new();
        encode(universe, &list, &mut buf);
        let mut cur = Cursor::new(universe, list.len(), &buf);
        assert_eq!(cur.next_geq(universe / 4), Some(universe / 2));
    }
}
