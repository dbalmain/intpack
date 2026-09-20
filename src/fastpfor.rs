//! FastPFor (Lemire & Boytsov §6): pages of 128-int blocks. Per block, pick a
//! bit width `b` minimising `128·b + exceptions·(maxbits − b)` plus 8 bits per
//! exception (its position) and 8 bits to store `maxbits`; pack the low `b`
//! bits, and collect exception high-parts into per-width arrays, themselves
//! bit-packed. A `VariableByte` tail covers `n % 128`.
//!
//! The on-disk layout is ours; the cost model matches the paper / the
//! `fastpfor` crate, so density comes out the same. The block packer is a
//! private scalar kernel — it will be replaced by the shared bitpack kernel.

use crate::delta;
use crate::vbyte;

const BLOCK: usize = 128;
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

/// Pack 32 integers using `width` bits each, LSB-first.
fn pack32(width: u32, src: &[u32], out: &mut Vec<u8>) {
    debug_assert!(src.len() >= 32);
    if width == 0 {
        return;
    }
    let mask = if width >= 32 {
        u64::from(u32::MAX)
    } else {
        (1u64 << width) - 1
    };
    let mut acc = 0u64;
    let mut have = 0u32;
    for &v in &src[..32] {
        acc |= (u64::from(v) & mask) << have;
        have += width;
        if have >= 32 {
            write_u32(out, acc as u32);
            acc >>= 32;
            have -= 32;
        }
    }
    if have > 0 {
        write_u32(out, acc as u32);
    }
}

fn unpack32(width: u32, buf: &[u8], pos: &mut usize, dst: &mut [u32]) {
    debug_assert!(dst.len() >= 32);
    if width == 0 {
        dst[..32].fill(0);
        return;
    }
    let mask = if width >= 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    };
    let mut acc = 0u64;
    let mut have = 0u32;
    for slot in &mut dst[..32] {
        if have < width {
            acc |= u64::from(read_u32(buf, pos)) << have;
            have += 32;
        }
        *slot = (acc as u32) & mask;
        acc >>= width;
        have -= width;
    }
}

/// Pack 128 integers at `width` bits. Temporary: will be replaced by the
/// shared bitpack kernel.
fn pack(width: u32, src: &[u32; 128], out: &mut Vec<u8>) {
    for chunk in src.chunks_exact(32) {
        pack32(width, chunk, out);
    }
}

/// Unpack 128 integers at `width` bits. Temporary: will be replaced by the
/// shared bitpack kernel.
fn unpack(width: u32, buf: &[u8], pos: &mut usize, dst: &mut [u32]) {
    debug_assert!(dst.len() >= BLOCK);
    match width {
        0 => dst[..BLOCK].fill(0),
        1 => {
            for slot_chunk in dst[..BLOCK].chunks_exact_mut(32) {
                let word = read_u32(buf, pos);
                for (bit, slot) in slot_chunk.iter_mut().enumerate() {
                    *slot = (word >> bit) & 1;
                }
            }
        }
        8 => {
            for (slot, &b) in dst[..BLOCK].iter_mut().zip(&buf[*pos..*pos + BLOCK]) {
                *slot = u32::from(b);
            }
            *pos += BLOCK;
        }
        16 => {
            for (slot, chunk) in dst[..BLOCK]
                .iter_mut()
                .zip(buf[*pos..*pos + BLOCK * 2].chunks_exact(2))
            {
                *slot = u32::from(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
            *pos += BLOCK * 2;
        }
        32 => {
            for slot in &mut dst[..BLOCK] {
                *slot = read_u32(buf, pos);
            }
        }
        _ => {
            for chunk in dst[..BLOCK].chunks_exact_mut(32) {
                unpack32(width, buf, pos, chunk);
            }
        }
    }
}

fn pack_n(width: u32, src: &[u32], out: &mut Vec<u8>) {
    if width == 0 || src.is_empty() {
        return;
    }
    let mut i = 0;
    while i + 32 <= src.len() {
        pack32(width, &src[i..], out);
        i += 32;
    }
    if i < src.len() {
        let mut tmp = [0u32; 32];
        tmp[..src.len() - i].copy_from_slice(&src[i..]);
        let start = out.len();
        pack32(width, &tmp, out);
        let keep = ((src.len() - i) * width as usize).div_ceil(32) * 4;
        out.truncate(start + keep);
    }
}

fn unpack_n(width: u32, count: usize, buf: &[u8], pos: &mut usize) -> Vec<u32> {
    let mut out = vec![0u32; count];
    if width == 0 || count == 0 {
        return out;
    }
    let mut i = 0;
    while i + 32 <= count {
        unpack32(width, buf, pos, &mut out[i..]);
        i += 32;
    }
    if i < count {
        let bytes = ((count - i) * width as usize).div_ceil(32) * 4;
        let mut padded = [0u8; 128];
        padded[..bytes].copy_from_slice(&buf[*pos..*pos + bytes]);
        *pos += bytes;
        let mut tmp = [0u32; 32];
        let mut p = 0;
        unpack32(width, &padded, &mut p, &mut tmp);
        out[i..].copy_from_slice(&tmp[..count - i]);
    }
    out
}

/// Choose `b` minimising the paper's cost. Returns `(b, nexcept, maxbits)`.
fn best_b(block: &[u32; 128]) -> (u8, u8, u8) {
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
        pack(u32::from(b), &block, out);
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
            pack_n(k as u32, slot, out);
        }
    }
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
    let mut except: [Vec<u32>; 33] = std::array::from_fn(|_| Vec::new());
    for (k, slot) in except.iter_mut().enumerate().skip(2) {
        if bitmap & (1 << (k - 1)) != 0 {
            let count = read_u32(buf, pos) as usize;
            *slot = unpack_n(k as u32, count, buf, pos);
        }
    }

    let mut packed_pos = 0;
    let mut meta_pos = 0;
    let mut except_i = [0usize; 33];
    let n_blocks = n / BLOCK;
    for b_i in 0..n_blocks {
        let width = u32::from(meta[meta_pos]);
        meta_pos += 1;
        let nexcept = usize::from(meta[meta_pos]);
        meta_pos += 1;
        let off = b_i * BLOCK;
        unpack(width, packed, &mut packed_pos, &mut dst[off..]);
        if nexcept > 0 {
            let maxb = meta[meta_pos];
            meta_pos += 1;
            let diff = usize::from(maxb) - width as usize;
            for _ in 0..nexcept {
                let idx = usize::from(meta[meta_pos]);
                meta_pos += 1;
                if diff == 1 {
                    dst[off + idx] |= 1 << width;
                } else {
                    dst[off + idx] |= except[diff][except_i[diff]] << width;
                    except_i[diff] += 1;
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
    fn pack32_writes_exactly_width_words() {
        for width in 0..=32u32 {
            let src = [0x1234_5678u32; 32];
            let mut buf = Vec::new();
            pack32(width, &src, &mut buf);
            assert_eq!(buf.len(), 4 * width as usize, "width={width}");
        }
    }

    #[test]
    fn pack_n_roundtrip_counts() {
        for width in 1..=32u32 {
            for count in 1..=40 {
                let mask = if width >= 32 {
                    u32::MAX
                } else {
                    (1u32 << width) - 1
                };
                let src: Vec<u32> = (0..count)
                    .map(|i| (i as u32).wrapping_mul(7) & mask)
                    .collect();
                let mut buf = Vec::new();
                pack_n(width, &src, &mut buf);
                let mut pos = 0;
                let got = unpack_n(width, count, &buf, &mut pos);
                assert_eq!(pos, buf.len(), "width={width} count={count} leftover");
                assert_eq!(got, src, "width={width} count={count}");
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
    fn pack_unpack_widths() {
        for width in 0..=32 {
            let mut src = [0u32; 128];
            let mask = if width == 0 {
                0
            } else if width >= 32 {
                u32::MAX
            } else {
                (1u32 << width) - 1
            };
            for (i, slot) in src.iter_mut().enumerate() {
                *slot = (i as u32).wrapping_mul(0x9e37) & mask;
            }
            let mut buf = Vec::new();
            pack(width, &src, &mut buf);
            let mut dst = [0u32; 128];
            let mut pos = 0;
            unpack(width, &buf, &mut pos, &mut dst);
            assert_eq!(dst, src, "width={width}");
            assert_eq!(pos, buf.len());
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
            let mask = if width == 0 {
                0
            } else if width >= 32 {
                u32::MAX
            } else {
                (1u32 << width) - 1
            };
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
