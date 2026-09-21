//! FastPFor (Lemire & Boytsov §6): pages of 128-int blocks. Per block, pick a
//! bit width `b` minimising `128·b + exceptions·(maxbits − b)` plus 8 bits per
//! exception (its position) and 8 bits to store `maxbits`; pack the low `b`
//! bits through [`bitpack`], and collect exception high-parts into per-width
//! arrays, themselves bit-packed with [`bits`]. A `VariableByte` tail covers
//! `n % 128`.
//!
//! Unlike [`bp128`](crate::bp128)'s at-most-seven inline byte patches,
//! FastPFor chooses exception counts by a cost model and collects high parts
//! across a page. The on-disk layout is ours; the cost model matches the
//! paper / the `fastpfor` crate, so density comes out the same. The 128-int
//! payload is the shared bitpack kernel (four interleaved lanes). Exception
//! arrays are a sequential LSB-first bit stream, byte-aligned, the cold path.

use crate::bitpack::{self, BLOCK};
use crate::{bits, delta, vbyte};

const PAGE: usize = 65536;
const OVERHEAD_OF_EACH_EXCEPT: u32 = 8;

fn bits_of(v: u32) -> usize {
    32 - v.leading_zeros() as usize
}

fn write_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn read_u32(buf: &[u8], pos: &mut usize) -> u32 {
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&buf[*pos..*pos + 4]);
    *pos += 4;
    u32::from_le_bytes(bytes)
}

fn low_mask(width: u32) -> u32 {
    if width >= 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    }
}

/// Choose `b` minimising the paper's cost. Returns `(b, nexcept, maxbits)`.
fn best_b(block: &[u32; BLOCK]) -> (u8, u8, u8) {
    let mut freqs = [0u32; 33];
    for &v in block {
        freqs[bits_of(v)] += 1;
    }
    let mut optimal_bits = 32u8;
    while freqs[usize::from(optimal_bits)] == 0 {
        optimal_bits -= 1;
    }
    let max_bits = optimal_bits;
    let mut best_cost = u32::from(optimal_bits) * BLOCK as u32;
    let mut exception_count = 0u8;
    let mut num_exceptions = 0u32;
    for bits in (0..optimal_bits).rev() {
        num_exceptions += freqs[usize::from(bits) + 1];
        if num_exceptions == BLOCK as u32 {
            break;
        }
        let diff = u32::from(max_bits - bits);
        let mut cost = num_exceptions * OVERHEAD_OF_EACH_EXCEPT
            + num_exceptions * diff
            + u32::from(bits) * BLOCK as u32
            + 8;
        if diff == 1 {
            cost -= num_exceptions;
        }
        if cost < best_cost {
            best_cost = cost;
            optimal_bits = bits;
            exception_count = num_exceptions as u8;
        }
    }
    (optimal_bits, exception_count, max_bits)
}

fn pack_block(width: u32, block: &mut [u32; BLOCK], out: &mut Vec<u8>) {
    if width < 32 {
        let m = low_mask(width);
        for v in &mut *block {
            *v &= m;
        }
    }
    let start = out.len();
    out.resize(start + bitpack::packed_len(width), 0);
    bitpack::pack(width, block, &mut out[start..]);
}

fn pack_except(width: u32, src: &[u32], out: &mut Vec<u8>) {
    let mut w = bits::Writer::new(out);
    for &v in src {
        w.put(u64::from(v), width);
    }
    w.align();
}

fn encode_page(page: &[u32], out: &mut Vec<u8>) {
    debug_assert!(!page.is_empty() && page.len().is_multiple_of(BLOCK));

    let packed_len_at = out.len();
    write_u32(out, 0);
    let packed_start = out.len();

    let mut meta = Vec::new();
    let mut except: [Vec<u32>; 33] = std::array::from_fn(|_| Vec::new());

    for raw in page.chunks_exact(BLOCK) {
        let mut block = [0u32; BLOCK];
        block.copy_from_slice(raw);
        let (b, nexcept, maxb) = best_b(&block);
        meta.push(b);
        meta.push(nexcept);
        if nexcept > 0 {
            meta.push(maxb);
            let diff = usize::from(maxb - b);
            for (k, &v) in block.iter().enumerate() {
                if v >> b != 0 {
                    meta.push(k as u8);
                    if diff != 1 {
                        except[diff].push(v >> b);
                    }
                }
            }
        }
        pack_block(u32::from(b), &mut block, out);
    }

    let packed_len = (out.len() - packed_start) as u32;
    out[packed_len_at..packed_len_at + 4].copy_from_slice(&packed_len.to_le_bytes());

    write_u32(out, meta.len() as u32);
    out.extend_from_slice(&meta);
    // Pad the metadata itself to a multiple of 4, not `out.len()`: a list
    // may be appended after an unaligned vbyte tail.
    let pad = meta.len().next_multiple_of(4) - meta.len();
    out.extend(std::iter::repeat_n(0, pad));

    let mut bitmap = 0u32;
    for (k, slot) in except.iter().enumerate().skip(2) {
        if !slot.is_empty() {
            bitmap |= 1 << (k - 1);
        }
    }
    write_u32(out, bitmap);
    for (k, slot) in except.iter().enumerate().skip(2) {
        if !slot.is_empty() {
            write_u32(out, slot.len() as u32);
            pack_except(k as u32, slot, out);
        }
    }
}

fn except_bytes(width: usize, count: usize) -> usize {
    (count * width).div_ceil(8)
}

fn decode_page(n: usize, buf: &[u8], pos: &mut usize, dst: &mut [u32]) {
    debug_assert!(n.is_multiple_of(BLOCK) && dst.len() >= n);
    let packed_len = read_u32(buf, pos) as usize;
    let packed_start = *pos;
    *pos += packed_len;
    let packed = &buf[packed_start..packed_start + packed_len];

    let meta_len = read_u32(buf, pos) as usize;
    let meta_start = *pos;
    let meta = &buf[meta_start..meta_start + meta_len];
    *pos = meta_start + meta_len.next_multiple_of(4);

    let bitmap = read_u32(buf, pos);
    let mut except_at = [0usize; 33];
    for (k, at) in except_at.iter_mut().enumerate().skip(2) {
        if bitmap & (1 << (k - 1)) != 0 {
            let count = read_u32(buf, pos) as usize;
            *at = *pos * 8;
            *pos += except_bytes(k, count);
        }
    }
    let bits = bits::Reader::new(buf);

    let mut packed_pos = 0;
    let mut meta_pos = 0;
    let (blocks, _) = dst[..n].as_chunks_mut::<BLOCK>();
    for dst_block in blocks {
        let width = u32::from(meta[meta_pos]);
        meta_pos += 1;
        let nexcept = usize::from(meta[meta_pos]);
        meta_pos += 1;
        let n_bytes = bitpack::packed_len(width);
        bitpack::unpack(width, &packed[packed_pos..packed_pos + n_bytes], dst_block);
        packed_pos += n_bytes;
        if nexcept > 0 {
            let maxb = meta[meta_pos];
            meta_pos += 1;
            let diff = usize::from(maxb) - width as usize;
            for _ in 0..nexcept {
                let idx = usize::from(meta[meta_pos]);
                meta_pos += 1;
                if diff == 1 {
                    dst_block[idx] |= 1u32 << width;
                } else {
                    let hi = bits.get(except_at[diff], diff as u32) as u32;
                    except_at[diff] += diff;
                    dst_block[idx] |= hi << width;
                }
            }
        }
    }
}

/// Encode an unsorted list.
pub fn encode(list: &[u32], out: &mut Vec<u8>) {
    let n_full = (list.len() / BLOCK) * BLOCK;
    write_u32(out, n_full as u32);
    let mut i = 0;
    while i < n_full {
        let end = (i + PAGE).min(n_full);
        encode_page(&list[i..end], out);
        i = end;
    }
    for &v in &list[n_full..] {
        vbyte::put(v, out);
    }
}

/// Decode `n` unsorted values, appending to `out`.
pub fn decode(n: usize, buf: &[u8], out: &mut Vec<u32>) {
    if n == 0 {
        return;
    }
    let start = out.len();
    out.resize(start + n, 0);
    let dst = &mut out[start..];
    let mut pos = 0;
    let n_full = read_u32(buf, &mut pos) as usize;
    let mut i = 0;
    while i < n_full {
        let end = (i + PAGE).min(n_full);
        decode_page(end - i, buf, &mut pos, &mut dst[i..end]);
        i = end;
    }
    for slot in &mut dst[n_full..] {
        *slot = vbyte::take(buf, &mut pos);
    }
}

/// Encode a strictly increasing list as deltas then FastPFor.
pub fn encode_sorted(list: &[u32], out: &mut Vec<u8>) {
    let mut gaps = Vec::with_capacity(list.len());
    delta::encode(None, list, &mut gaps);
    encode(&gaps, out);
}

/// Decode `n` sorted values, appending to `out`.
pub fn decode_sorted(n: usize, buf: &[u8], out: &mut Vec<u32>) {
    let start = out.len();
    decode(n, buf, out);
    delta::decode(None, &mut out[start..]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn roundtrip_unsorted(list: &[u32]) {
        let mut buf = Vec::new();
        encode(list, &mut buf);
        let mut out = Vec::new();
        decode(list.len(), &buf, &mut out);
        assert_eq!(out, list, "unsorted n={}", list.len());
    }

    fn roundtrip_sorted(list: &[u32]) {
        let mut buf = Vec::new();
        encode_sorted(list, &mut buf);
        let mut out = Vec::new();
        decode_sorted(list.len(), &buf, &mut out);
        assert_eq!(out, list, "sorted n={}", list.len());
    }

    fn consecutive(n: u32) -> Vec<u32> {
        (0..n).collect()
    }

    #[test]
    fn roundtrip_lengths() {
        for n in [0, 1, 3, 4, 5, 127, 128, 129, 1000] {
            roundtrip_unsorted(&consecutive(n));
            roundtrip_sorted(&consecutive(n));
        }
    }

    #[test]
    fn dense_run_all_gaps_zero() {
        roundtrip_sorted(&consecutive(1000));
        roundtrip_unsorted(&vec![0; 256]);
    }

    #[test]
    fn max_value() {
        roundtrip_unsorted(&[u32::MAX]);
        roundtrip_unsorted(&[0, u32::MAX, 1]);
        let mut block = vec![0u32; 128];
        block[0] = u32::MAX;
        block[127] = u32::MAX;
        roundtrip_unsorted(&block);
        roundtrip_sorted(&[0, 1, u32::MAX]);
        roundtrip_sorted(&[u32::MAX]);
    }

    #[test]
    fn except_arrays_roundtrip_counts() {
        for width in 2..=32u32 {
            for count in 1..=40 {
                let mask = low_mask(width);
                let src: Vec<u32> = (0..count)
                    .map(|i| (i as u32).wrapping_mul(7) & mask)
                    .collect();
                let mut buf = Vec::new();
                pack_except(width, &src, &mut buf);
                assert_eq!(buf.len(), except_bytes(width as usize, count));
                let r = bits::Reader::new(&buf);
                for (i, &want) in src.iter().enumerate() {
                    assert_eq!(
                        r.get(i * width as usize, width) as u32,
                        want,
                        "width={width} count={count} i={i}"
                    );
                }
            }
        }
    }

    #[test]
    fn full_page_random() {
        let mut rng = rand::rng();
        let list: Vec<u32> = (0..PAGE).map(|_| rng.random_range(0..64)).collect();
        roundtrip_unsorted(&list);
        let list: Vec<u32> = (0..PAGE).map(|_| rng.random::<u32>()).collect();
        roundtrip_unsorted(&list);
    }

    #[test]
    fn one_block_every_width() {
        for width in 0..=32 {
            let mask = low_mask(width);
            let src: Vec<u32> = (0..128)
                .map(|i| (i as u32).wrapping_mul(0x9e37) & mask)
                .collect();
            roundtrip_unsorted(&src);
        }
    }

    #[test]
    fn random_widths_per_block() {
        let mut rng = rand::rng();
        let mut list = Vec::new();
        // Blocks with no exceptions, mixed, and many exceptions.
        for &(width, nexcept) in &[
            (0u32, 0usize),
            (3, 0),
            (4, 3),
            (7, 20),
            (8, 0),
            (12, 40),
            (16, 5),
            (20, 60),
            (31, 2),
        ] {
            let mask = low_mask(width);
            let mut block = [0u32; 128];
            for slot in &mut block {
                *slot = rng.random::<u32>() & mask;
            }
            for _ in 0..nexcept {
                let i = rng.random_range(0..128);
                let room = 32 - width;
                if room == 0 {
                    continue;
                }
                let extra = rng.random_range(1..=room.min(8));
                let hi = if extra == 32 {
                    rng.random::<u32>()
                } else {
                    rng.random::<u32>() & ((1u32 << extra) - 1)
                };
                block[i] |= hi << width;
            }
            list.extend_from_slice(&block);
        }
        // Remainder that is not a full block.
        list.extend_from_slice(&[1, 2, 3, 1 << 20, u32::MAX]);
        roundtrip_unsorted(&list);
    }

    #[test]
    fn encode_appended_after_unaligned_tail() {
        // One leftover vbyte byte makes the next list start unaligned.
        let a: Vec<u32> = (0..129).collect();
        let b: Vec<u32> = (0..1000).map(|i| i * 10).collect();
        let mut out = Vec::new();
        encode_sorted(&a, &mut out);
        assert!(!out.len().is_multiple_of(4), "precondition: unaligned join");
        let mid = out.len();
        encode_sorted(&b, &mut out);
        let mut got = Vec::new();
        decode_sorted(b.len(), &out[mid..], &mut got);
        assert_eq!(got, b);
    }

    #[test]
    fn two_pages() {
        // 65536 + 128 forces a second page.
        let list: Vec<u32> = (0..(PAGE as u32 + 128)).map(|i| i % 17).collect();
        roundtrip_unsorted(&list);
    }

    #[test]
    fn long_second_page_small_values() {
        // Matches uniform-d0.1 shape: ~100k ints, two pages, small gaps.
        let n = 100_736u32;
        let list: Vec<u32> = (0..n).map(|i| i % 32).collect();
        roundtrip_unsorted(&list);
        let sorted: Vec<u32> = (0..n).map(|i| i * 10).collect();
        roundtrip_sorted(&sorted);
    }
}
