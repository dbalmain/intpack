//! Fixed-128 partitioned Elias-Fano over strictly increasing `u32` lists.
//!
//! The upper level is a plain Elias-Fano sequence of partition maxima; a
//! packed two-bit directory selects all-ones, bitmap, or Elias-Fano storage
//! per partition; a fixed-width packed offset directory addresses the
//! byte-aligned payloads; and, new versus the reference `pef`, the upper
//! level's select samples let a fresh seek jump straight to the target
//! partition instead of binary-searching with a linear select each time.
//! Elias-Fano partitions reuse [`elias_fano`]'s low/high primitives on a
//! sub-slice rather than carrying a second copy of them.
//!
//! Byte layout:
//!
//! ```text
//! [upper_l: u8][offset_width: u8]
//! [upper lows][upper highs]                 partition maxima, Elias-Fano
//! [types][packed offsets]                   2-bit directory + offset directory
//! [payload]                                 per partition: all-ones = nothing;
//!                                           bitmap = bitmap; EF = [lows][highs]
//! [upper select samples]                    select positions, one u32 per 256 partitions
//! ```
//!
//! `n` and `universe` are stored out of band by the caller. Empty lists
//! encode to an empty buffer.

use crate::bits::{Reader, Writer};
use crate::elias_fano::{self, Layout as EfLayout, SELECT_Q};

const PARTITION_SIZE: usize = 128;
const TYPE_ALL_ONES: u8 = 0;
const TYPE_BITMAP: u8 = 1;
const TYPE_ELIAS_FANO: u8 = 2;

/// Append the partitioned encoding of `list` (values in `[0, universe)`) to
/// `out`.
pub fn encode(universe: u32, list: &[u32], out: &mut Vec<u8>) {
    if list.is_empty() {
        return;
    }
    let partition_count = list.len().div_ceil(PARTITION_SIZE);
    let upper_l = elias_fano::low_width(universe, partition_count);
    let upper_bounds: Vec<u32> = list
        .chunks(PARTITION_SIZE)
        .map(|p| p[p.len() - 1])
        .collect();

    let mut types = vec![0u8; partition_count.div_ceil(4)];
    let mut offsets: Vec<u64> = Vec::with_capacity(partition_count);
    let mut payload = Vec::new();
    let mut start = 0u64;
    for (pi, partition) in list.chunks(PARTITION_SIZE).enumerate() {
        offsets.push(payload.len() as u64);
        let hi = u64::from(partition[partition.len() - 1]);
        let local_universe = hi + 1 - start;
        let ef_l = elias_fano::low_width(local_universe as u32, partition.len());
        let bitmap_bytes = (local_universe as usize).div_ceil(8);
        let ef_bytes = (partition.len() * ef_l as usize).div_ceil(8)
            + (partition.len() + (local_universe >> ef_l) as usize).div_ceil(8);
        let ptype = if local_universe == partition.len() as u64 {
            TYPE_ALL_ONES
        } else if bitmap_bytes <= ef_bytes {
            TYPE_BITMAP
        } else {
            TYPE_ELIAS_FANO
        };
        types[pi / 4] |= ptype << ((pi % 4) * 2);
        match ptype {
            TYPE_ALL_ONES => {}
            TYPE_BITMAP => encode_bitmap(partition, start, local_universe as usize, &mut payload),
            TYPE_ELIAS_FANO => {
                elias_fano::encode_parts(
                    local_universe as u32,
                    ef_l,
                    start as u32,
                    partition,
                    &mut payload,
                );
            }
            _ => unreachable!(),
        }
        start = hi + 1;
    }

    let offset_width = bits_required(payload.len() as u64);
    out.push(upper_l as u8);
    out.push(offset_width);
    elias_fano::encode_parts(universe, upper_l, 0, &upper_bounds, out);
    out.extend_from_slice(&types);
    let mut writer = Writer::new(out);
    for offset in &offsets {
        writer.put(*offset, u32::from(offset_width));
    }
    writer.align();
    out.extend_from_slice(&payload);
    for (i, &b) in upper_bounds.iter().enumerate().step_by(SELECT_Q) {
        let pos = (u64::from(b) >> upper_l) as usize + i;
        out.extend_from_slice(&(pos as u32).to_le_bytes());
    }
    // Trailing marker so `aux_bytes` can report the index overhead without
    // needing `universe`.
    let aux = 2
        + partition_count.div_ceil(4)
        + (partition_count * u32::from(offset_width) as usize).div_ceil(8)
        + partition_count.div_ceil(SELECT_Q) * 4;
    out.extend_from_slice(&(aux as u32).to_le_bytes());
}

/// Reconstruct exactly the `n` values `encode` produced, appending to `out`.
pub fn decode(universe: u32, n: usize, buf: &[u8], out: &mut Vec<u32>) {
    if n == 0 {
        return;
    }
    let layout = Layout::new(universe, n, buf);
    for pi in 0..layout.partition_count {
        let partition = layout.partition(pi);
        match &partition.data {
            PartitionData::AllOnes => {
                for i in 0..partition.len {
                    out.push(partition.start + i as u32);
                }
            }
            PartitionData::Bitmap(bits) => {
                let mut next = 0usize;
                for _ in 0..partition.len {
                    let Some(bit) = find_next_one(bits, partition.universe, next) else {
                        break;
                    };
                    out.push(partition.start + bit as u32);
                    next = bit + 1;
                }
            }
            PartitionData::EliasFano { ef, payload } => {
                elias_fano::decode_parts(ef.l, partition.len, payload, partition.start, out);
            }
        }
    }
}

/// Bytes the select samples and the directory occupy; the offset directory
/// width is not knowable without the buffer, so this under-reports by it.
/// Callers that hold the buffer should use [`aux_bytes`].
pub fn aux_len(_universe: u32, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let partition_count = n.div_ceil(PARTITION_SIZE);
    2 + partition_count.div_ceil(4) + partition_count.div_ceil(SELECT_Q) * 4
}

/// Full index overhead from an encoded buffer: header + type directory +
/// offset directory + upper select samples.
pub fn aux_bytes(n: usize, buf: &[u8]) -> usize {
    if n == 0 {
        return 0;
    }
    let b = &buf[buf.len() - 4..];
    u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize
}

/// The `i`-th value by ordinal.
pub fn get(universe: u32, n: usize, buf: &[u8], i: usize) -> u32 {
    assert!(i < n, "pef::get({i}) out of range (n={n})");
    let layout = Layout::new(universe, n, buf);
    let Some(value) = layout
        .partition(i / PARTITION_SIZE)
        .value_at(i % PARTITION_SIZE)
    else {
        unreachable!("i < n and the list round-trips, so the element exists");
    };
    value
}

fn bits_required(value: u64) -> u8 {
    (u64::BITS - value.leading_zeros()) as u8
}

fn encode_bitmap(values: &[u32], start: u64, universe: usize, out: &mut Vec<u8>) {
    let offset = out.len();
    out.resize(offset + universe.div_ceil(8), 0);
    for &value in values {
        set_bit(&mut out[offset..], (u64::from(value) - start) as usize);
    }
}

fn set_bit(bytes: &mut [u8], bit: usize) {
    bytes[bit / 8] |= 1 << (bit % 8);
}

fn word_from_chunk(chunk: &[u8]) -> u64 {
    let mut bytes = [0u8; 8];
    bytes[..chunk.len()].copy_from_slice(chunk);
    u64::from_le_bytes(bytes)
}

fn trailing_mask(bits: usize) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

/// The `rank`-th set bit (0-indexed) in a bitmap.
fn find_nth_one(bytes: &[u8], bit_len: usize, rank: usize) -> Option<usize> {
    let mut remaining = rank;
    for (word_index, chunk) in bytes.chunks(8).enumerate() {
        let mut word = word_from_chunk(chunk);
        if word_index * 64 + 64 > bit_len {
            word &= trailing_mask(bit_len.saturating_sub(word_index * 64));
        }
        let ones = word.count_ones() as usize;
        if remaining < ones {
            let mut w = word;
            for _ in 0..remaining {
                w &= w - 1;
            }
            return Some(word_index * 64 + w.trailing_zeros() as usize);
        }
        remaining -= ones;
    }
    None
}

/// The next set bit at or after `from`, or `None`.
fn find_next_one(bytes: &[u8], bit_len: usize, from: usize) -> Option<usize> {
    if from >= bit_len {
        return None;
    }
    let mut word_index = from / 64;
    let mut first = true;
    while word_index * 64 < bit_len {
        let byte_start = word_index * 8;
        let byte_end = (byte_start + 8).min(bytes.len());
        let mut word = word_from_chunk(&bytes[byte_start..byte_end]);
        if first {
            word &= u64::MAX << (from % 64);
            first = false;
        }
        if word_index * 64 + 64 > bit_len {
            word &= trailing_mask(bit_len - word_index * 64);
        }
        if word != 0 {
            return Some(word_index * 64 + word.trailing_zeros() as usize);
        }
        word_index += 1;
    }
    None
}

/// Set bits strictly before `bit`.
fn count_ones_before(bytes: &[u8], bit: usize) -> usize {
    let full_bytes = bit / 8;
    let mut count: usize = bytes[..full_bytes]
        .iter()
        .map(|byte| byte.count_ones() as usize)
        .sum();
    if !bit.is_multiple_of(8) {
        count += (bytes[full_bytes] & ((1 << (bit % 8)) - 1)).count_ones() as usize;
    }
    count
}

/// Byte layout of one encoded list, everything in place.
struct Layout<'a> {
    buf: &'a [u8],
    n: usize,
    partition_count: usize,
    upper_l: u32,
    upper_lows_bit: usize,
    types_start: usize,
    offsets_start: usize,
    offset_width: u32,
    payload_start: usize,
    samples_start: usize,
    universe: u32,
}

impl<'a> Layout<'a> {
    fn new(universe: u32, n: usize, buf: &'a [u8]) -> Self {
        let partition_count = n.div_ceil(PARTITION_SIZE);
        if n == 0 {
            return Self {
                buf,
                n,
                partition_count,
                upper_l: 0,
                upper_lows_bit: 0,
                types_start: 0,
                offsets_start: 0,
                offset_width: 0,
                payload_start: 0,
                samples_start: 0,
                universe: 0,
            };
        }
        let upper_l = u32::from(buf[0]);
        let offset_width = u32::from(buf[1]);
        let upper_lows_bit = 16;
        let upper_lows_bytes = (partition_count * upper_l as usize).div_ceil(8);
        let upper_highs_start = 2 + upper_lows_bytes;
        let upper_highs_bits = partition_count + (u64::from(universe) >> upper_l) as usize;
        let types_start = upper_highs_start + upper_highs_bits.div_ceil(8);
        let offsets_start = types_start + partition_count.div_ceil(4);
        let offsets_bytes = (partition_count * offset_width as usize).div_ceil(8);
        let payload_start = offsets_start + offsets_bytes;
        let select1_bytes = partition_count.div_ceil(SELECT_Q) * 4;
        let samples_start = buf.len() - 4 - select1_bytes;
        Self {
            buf,
            n,
            partition_count,
            upper_l,
            upper_lows_bit,
            types_start,
            offsets_start,
            offset_width,
            payload_start,
            samples_start,
            universe,
        }
    }

    /// Upper-level EF with its select samples, for partition seeks.
    fn upper_ef(&self) -> EfLayout<'a> {
        EfLayout::from_parts_at(
            self.upper_l,
            self.partition_count,
            self.universe,
            self.buf,
            self.upper_lows_bit,
            crate::elias_fano::SampleIndex {
                select1: self.samples_start,
                select0_super: 0,
                select0_sub: 0,
            },
        )
    }

    fn upper_bound(&self, partition_index: usize) -> u32 {
        let ef = self.upper_ef();
        let pos = ef.select_one(partition_index);
        ef.value_at(partition_index, pos)
    }

    /// First partition whose maximum is `>= target`, via the sampled upper
    /// EF — the fresh-seek path that the reference `pef` lacks.
    fn partition_for(&self, target: u32) -> Option<usize> {
        let ef = self.upper_ef();
        let (index, pos) = ef.land(target)?;
        ef.scan(index, pos, target).map(|(i, _, _)| i)
    }

    fn partition(&self, partition_index: usize) -> Partition<'a> {
        count_partition_decode();
        let ef = self.upper_ef();
        let (start, hi) = if partition_index == 0 {
            (0u32, self.upper_bound(0))
        } else {
            let previous = ef.select_one(partition_index - 1);
            let previous_bound = ef.value_at(partition_index - 1, previous);
            let Some(position) = ef.next_one_after(previous) else {
                return Partition {
                    start: previous_bound + 1,
                    len: 0,
                    universe: 0,
                    data: PartitionData::AllOnes,
                };
            };
            (previous_bound + 1, ef.value_at(partition_index, position))
        };
        let universe = (u64::from(hi) + 1 - u64::from(start)) as usize;
        let len = (self.n - partition_index * PARTITION_SIZE).min(PARTITION_SIZE);
        let ptype =
            (self.buf[self.types_start + partition_index / 4] >> ((partition_index % 4) * 2)) & 3;
        let payload_offset = read_bits(
            &self.buf[self.offsets_start..self.payload_start],
            partition_index * self.offset_width as usize,
            self.offset_width,
        ) as usize;
        let payload = &self.buf[self.payload_start + payload_offset..];
        let data = match ptype {
            TYPE_ALL_ONES => PartitionData::AllOnes,
            TYPE_BITMAP => PartitionData::Bitmap(&payload[..universe.div_ceil(8)]),
            TYPE_ELIAS_FANO => {
                let ef_l = elias_fano::low_width(universe as u32, len);
                let ef = EfLayout::from_parts(ef_l, len, universe as u32, payload);
                PartitionData::EliasFano { ef, payload }
            }
            _ => unreachable!(),
        };
        Partition {
            start,
            len,
            universe,
            data,
        }
    }
}

fn read_bits(bytes: &[u8], bit_offset: usize, width: u32) -> u64 {
    Reader::new(bytes).get(bit_offset, width)
}

enum PartitionData<'a> {
    AllOnes,
    Bitmap(&'a [u8]),
    EliasFano { ef: EfLayout<'a>, payload: &'a [u8] },
}

struct Partition<'a> {
    start: u32,
    len: usize,
    universe: usize,
    data: PartitionData<'a>,
}

impl Partition<'_> {
    fn value_at(&self, index: usize) -> Option<u32> {
        if index >= self.len {
            return None;
        }
        let local = match &self.data {
            PartitionData::AllOnes => index as u32,
            PartitionData::Bitmap(bits) => find_nth_one(bits, self.universe, index)? as u32,
            PartitionData::EliasFano { ef, .. } => {
                let pos = ef.select_one(index);
                ef.value_at(index, pos)
            }
        };
        Some(self.start + local)
    }

    fn first_geq(&self, target: u32, from_index: usize) -> Option<Position> {
        match &self.data {
            PartitionData::AllOnes => {
                let local = (target as usize).max(from_index);
                (local < self.len).then_some(Position {
                    index: local,
                    local: local as u32,
                })
            }
            PartitionData::Bitmap(bits) => {
                let local = find_next_one(bits, self.universe, target as usize)?;
                let index = count_ones_before(bits, local);
                (index >= from_index).then_some(Position {
                    index,
                    local: local as u32,
                })
            }
            PartitionData::EliasFano { ef, .. } => {
                let (index, _, local) = ef.scan(from_index, ef.select_one(from_index), target)?;
                Some(Position { index, local })
            }
        }
    }

    fn next_after(&self, current: &Position) -> Option<Position> {
        let index = current.index + 1;
        if index >= self.len {
            return None;
        }
        match &self.data {
            PartitionData::AllOnes => Some(Position {
                index,
                local: index as u32,
            }),
            PartitionData::Bitmap(bits) => {
                let local = find_next_one(bits, self.universe, current.local as usize + 1)?;
                Some(Position {
                    index,
                    local: local as u32,
                })
            }
            PartitionData::EliasFano { ef, .. } => {
                let pos = ef.next_one_after(ef.select_one(current.index))?;
                Some(Position {
                    index,
                    local: ef.value_at(index, pos),
                })
            }
        }
    }
}

/// One element's index and partition-local value.
struct Position {
    index: usize,
    local: u32,
}

/// Skip cursor: jumps to the target partition via the sampled upper EF,
/// decodes only that partition, and scans within it.
pub struct Cursor<'a> {
    layout: Layout<'a>,
    partition_index: usize,
    partition: Option<Partition<'a>>,
    position: Option<Position>,
    current: Option<u32>,
}

impl<'a> Cursor<'a> {
    pub fn new(universe: u32, n: usize, buf: &'a [u8]) -> Self {
        Self {
            layout: Layout::new(universe, n, buf),
            partition_index: usize::MAX,
            partition: None,
            position: None,
            current: None,
        }
    }

    fn load_partition(&mut self, partition_index: usize) {
        self.partition_index = partition_index;
        self.partition = Some(self.layout.partition(partition_index));
        self.position = None;
        self.current = None;
    }

    fn set_position(&mut self, position: Position) -> Option<u32> {
        let partition = self.partition.as_ref()?;
        let value = partition.start + position.local;
        self.position = Some(position);
        self.current = Some(value);
        Some(value)
    }

    fn exhaust(&mut self) -> Option<u32> {
        self.partition_index = self.layout.partition_count;
        self.partition = None;
        self.position = None;
        self.current = None;
        None
    }
}

impl crate::Cursor for Cursor<'_> {
    fn next_geq(&mut self, target: u32) -> Option<u32> {
        if self.current.is_some_and(|current| current >= target) {
            return self.current;
        }
        if self.layout.partition_count == 0 || self.partition_index == self.layout.partition_count {
            return None;
        }
        let target_past_partition = self
            .partition
            .as_ref()
            .is_none_or(|p| u64::from(p.start) + p.universe as u64 <= u64::from(target));
        if self.partition_index == usize::MAX || target_past_partition {
            let partition_index = self.layout.partition_for(target)?;
            self.load_partition(partition_index);
        }
        let partition = self.partition.as_ref()?;
        let local_target = target.saturating_sub(partition.start);
        let from_index = self.position.as_ref().map_or(0, |p| p.index);
        let position = partition.first_geq(local_target, from_index)?;
        self.set_position(position)
    }

    fn next(&mut self) -> Option<u32> {
        if self.layout.partition_count == 0 || self.partition_index == self.layout.partition_count {
            return None;
        }
        if self.partition_index == usize::MAX {
            self.load_partition(0);
            let position = self.partition.as_ref()?.first_geq(0, 0)?;
            return self.set_position(position);
        }
        if let (Some(partition), Some(position)) = (&self.partition, &self.position)
            && let Some(next) = partition.next_after(position)
        {
            return self.set_position(next);
        }
        let next_partition = self.partition_index + 1;
        if next_partition >= self.layout.partition_count {
            return self.exhaust();
        }
        self.load_partition(next_partition);
        let position = self.partition.as_ref()?.first_geq(0, 0)?;
        self.set_position(position)
    }
}

#[cfg(test)]
thread_local! {
    static PARTITION_DECODES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn count_partition_decode() {
    PARTITION_DECODES.with(|count| count.set(count.get() + 1));
}

#[cfg(not(test))]
fn count_partition_decode() {}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::cursor::Cursor as _;

    use super::*;

    fn enc(list: &[u32], universe: u32) -> Vec<u8> {
        let mut encoded = Vec::new();
        encode(universe, list, &mut encoded);
        encoded
    }

    fn roundtrip(list: &[u32], universe: u32) {
        let encoded = enc(list, universe);
        let mut decoded = Vec::new();
        decode(universe, list.len(), &encoded, &mut decoded);
        assert_eq!(decoded, list);
        for (index, &value) in list.iter().enumerate() {
            assert_eq!(get(universe, list.len(), &encoded, index), value);
        }
    }

    #[test]
    fn lengths_roundtrip() {
        let universe = 1 << 20;
        for n in [0, 1, 2, 64, 65, 128, 129, 1000] {
            let list: Vec<u32> = (0..n as u32).map(|i| i * 7 + 3).collect();
            roundtrip(&list, universe);
        }
    }

    #[test]
    fn edge_lists() {
        let universe = 1 << 20;
        roundtrip(&[0], universe);
        roundtrip(&[universe - 1], universe);
        roundtrip(&[0, universe - 1], universe);
    }

    #[test]
    fn dense() {
        let universe = 1 << 12;
        let list: Vec<u32> = (0..universe).collect();
        roundtrip(&list, universe);
    }

    #[test]
    fn max_universe() {
        let universe = u32::MAX - 1;
        roundtrip(&[0], universe);
        roundtrip(&[universe - 1], universe);
        roundtrip(&[0, universe - 1], universe);
        let list: Vec<u32> = (0..1000).map(|i| i * 1_000_000).collect();
        roundtrip(&list, universe);
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
        roundtrip(&list, universe);
    }

    #[test]
    fn all_partition_types() {
        let mut list: Vec<u32> = (0..=300).collect();
        list.extend((0..400).map(|index| 302 + index * 2));
        let sparse_start = list[list.len() - 1] + 10_000;
        list.extend((0..300).map(|index| sparse_start + index * 10_000));
        let universe = list[list.len() - 1] + 1;
        roundtrip(&list, universe);
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
        ];
        for list in lists {
            let encoded = enc(&list, universe);
            let mut cur = Cursor::new(universe, list.len(), &encoded);
            for (i, &v) in list.iter().enumerate() {
                let target = if i > 0 && list[i - 1] + 1 < v {
                    list[i - 1] + 1
                } else {
                    v
                };
                assert_eq!(cur.next_geq(target), Some(v), "next_geq({target}) != {v}");
                assert_eq!(cur.next_geq(v), Some(v), "re-seek moved off {v}");
            }
            assert_eq!(cur.next_geq(universe), None);
            let mut cur = Cursor::new(universe, list.len(), &encoded);
            for &v in &list {
                assert_eq!(cur.next(), Some(v));
            }
            assert_eq!(cur.next(), None);
        }
    }

    #[test]
    fn seek_lands_far_partition_and_decodes_one() {
        let list: Vec<u32> = (0..128_000).map(|value| value * 10).collect();
        let universe = list[list.len() - 1] + 1;
        let encoded = enc(&list, universe);
        let mut cursor = Cursor::new(universe, list.len(), &encoded);
        assert_eq!(cursor.next(), Some(0));
        PARTITION_DECODES.with(|count| count.set(0));
        assert_eq!(cursor.next_geq(list[900 * 128]), Some(list[900 * 128]));
        PARTITION_DECODES.with(|count| assert_eq!(count.get(), 1));
    }

    #[test]
    fn size_sanity() {
        let every_second: Vec<u32> = (0..100_000).step_by(2).collect();
        let encoded = enc(&every_second, 100_000);
        let bits_per_int = encoded.len() as f64 * 8.0 / every_second.len() as f64;
        assert!(bits_per_int <= 2.25, "{bits_per_int} bits/int");
    }
}
