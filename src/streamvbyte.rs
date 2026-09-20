//! Stream VByte (Lemire & Kurz): groups of four values, a 2-bit length tag
//! per value packed into a control byte, all control bytes first, then the
//! data bytes. A tail of `n % 4` values is a final partial group.
//!
//! Sorted lists use [`crate::delta`]. The scalar codec *is* the format; the
//! SSSE3 kernel behind it must agree bit-for-bit.

use crate::delta;

/// pshufb control: scatter packed bytes back into four LE u32s (unused lanes 0).
#[cfg(target_arch = "x86_64")]
const DECODE_SHUFFLE: [[u8; 16]; 256] = {
    let mut t = [[0x80u8; 16]; 256];
    let mut tag = 0;
    while tag < 256 {
        t[tag] = decode_shuffle_entry(tag as u8);
        tag += 1;
    }
    t
};

#[cfg(target_arch = "x86_64")]
const fn decode_shuffle_entry(tag: u8) -> [u8; 16] {
    let mut shuf = [0x80u8; 16];
    let mut src = 0u8;
    let mut i = 0;
    while i < 4 {
        let len = ((tag >> (2 * i)) & 3) + 1;
        let mut j = 0;
        while j < len {
            shuf[i * 4 + j as usize] = src;
            src += 1;
            j += 1;
        }
        i += 1;
    }
    shuf
}

#[inline]
fn tag_of(v: u32) -> u8 {
    (3 - (v.leading_zeros() / 8).min(3)) as u8
}

fn write_group(vals: &[u32], n: usize, data: &mut Vec<u8>) -> u8 {
    let mut tag = 0u8;
    for (i, &v) in vals[..n].iter().enumerate() {
        let t = tag_of(v);
        tag |= t << (2 * i);
        let nbytes = t as usize + 1;
        data.extend_from_slice(&v.to_le_bytes()[..nbytes]);
    }
    tag
}

fn read_group(tag: u8, n: usize, data: &[u8], pos: &mut usize, dst: &mut [u32]) {
    for (i, slot) in dst.iter_mut().take(n).enumerate() {
        let len = ((tag >> (2 * i)) & 3) as usize + 1;
        let mut bytes = [0u8; 4];
        bytes[..len].copy_from_slice(&data[*pos..*pos + len]);
        *slot = u32::from_le_bytes(bytes);
        *pos += len;
    }
}

fn encode_values(list: &[u32], out: &mut Vec<u8>) {
    let n = list.len();
    let n_groups = n.div_ceil(4);
    let tag_at = out.len();
    out.resize(tag_at + n_groups, 0);
    let mut g = 0;
    let mut i = 0;
    while i + 4 <= n {
        out[tag_at + g] = write_group(&list[i..], 4, out);
        g += 1;
        i += 4;
    }
    if i < n {
        out[tag_at + g] = write_group(&list[i..], n - i, out);
    }
}

fn decode_values_scalar(n: usize, buf: &[u8], out: &mut Vec<u32>) {
    let n_groups = n.div_ceil(4);
    let tags = &buf[..n_groups];
    let data = &buf[n_groups..];
    let start = out.len();
    out.resize(start + n, 0);
    let dst = &mut out[start..];
    let mut pos = 0;
    let mut i = 0;
    let mut g = 0;
    while i + 4 <= n {
        read_group(tags[g], 4, data, &mut pos, &mut dst[i..]);
        g += 1;
        i += 4;
    }
    if i < n {
        read_group(tags[g], n - i, data, &mut pos, &mut dst[i..]);
    }
}

/// Encode an unsorted list.
pub fn encode(list: &[u32], out: &mut Vec<u8>) {
    encode_values(list, out);
}

/// Decode `n` unsorted values, appending to `out`.
pub fn decode(n: usize, buf: &[u8], out: &mut Vec<u32>) {
    if n == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("ssse3") {
            let start = out.len();
            out.resize(start + n, 0);
            // Safety: SSSE3 was detected.
            unsafe {
                kernel::decode_ssse3(n, buf, &mut out[start..]);
            }
            return;
        }
    }
    decode_values_scalar(n, buf, out);
}

/// Encode a strictly increasing list as deltas then Stream VByte.
pub fn encode_sorted(list: &[u32], out: &mut Vec<u8>) {
    let n = list.len();
    let n_groups = n.div_ceil(4);
    let tag_at = out.len();
    out.resize(tag_at + n_groups, 0);
    let mut prev = None;
    let mut buf = [0u32; 4];
    let mut k = 0;
    let mut g = 0;
    for &v in list {
        buf[k] = delta::gap(prev, v);
        prev = Some(v);
        k += 1;
        if k == 4 {
            out[tag_at + g] = write_group(&buf, 4, out);
            g += 1;
            k = 0;
        }
    }
    if k > 0 {
        out[tag_at + g] = write_group(&buf, k, out);
    }
}

/// Decode `n` sorted values, appending to `out`.
pub fn decode_sorted(n: usize, buf: &[u8], out: &mut Vec<u32>) {
    if n == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("ssse3") {
            let start = out.len();
            out.resize(start + n, 0);
            // Safety: SSSE3 was detected.
            unsafe {
                kernel::decode_sorted_ssse3(n, buf, &mut out[start..]);
            }
            return;
        }
    }
    decode_sorted_scalar(n, buf, out);
}

fn decode_sorted_scalar(n: usize, buf: &[u8], out: &mut Vec<u32>) {
    let n_groups = n.div_ceil(4);
    let tags = &buf[..n_groups];
    let data = &buf[n_groups..];
    let start = out.len();
    out.resize(start + n, 0);
    let dst = &mut out[start..];
    let mut pos = 0;
    let mut tmp = [0u32; 4];
    read_group(tags[0], n.min(4), data, &mut pos, &mut tmp);
    let first_n = n.min(4);
    dst[0] = tmp[0];
    let mut prev = tmp[0];
    for i in 1..first_n {
        prev += tmp[i] + 1;
        dst[i] = prev;
    }
    let mut i = first_n;
    let mut g = 1;
    while i + 4 <= n {
        read_group(tags[g], 4, data, &mut pos, &mut tmp);
        for k in 0..4 {
            prev += tmp[k] + 1;
            dst[i + k] = prev;
        }
        g += 1;
        i += 4;
    }
    if i < n {
        read_group(tags[g], n - i, data, &mut pos, &mut tmp);
        for k in 0..(n - i) {
            prev += tmp[k] + 1;
            dst[i + k] = prev;
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod kernel {
    use super::DECODE_SHUFFLE;

    use std::arch::x86_64::{
        __m128i, _mm_add_epi32, _mm_bslli_si128, _mm_loadu_si128, _mm_set1_epi32, _mm_setr_epi32,
        _mm_shuffle_epi8, _mm_storeu_si128,
    };

    /// Length in bytes of the data stream for one control byte.
    const DATA_LEN: [u8; 256] = {
        let mut t = [0u8; 256];
        let mut tag = 0;
        while tag < 256 {
            t[tag] = (4 + (tag & 3) + ((tag >> 2) & 3) + ((tag >> 4) & 3) + ((tag >> 6) & 3)) as u8;
            tag += 1;
        }
        t
    };

    /// Load 16 data bytes, zero-padding if fewer than 16 remain.
    #[inline]
    unsafe fn load16(ptr: *const u8, remain: usize) -> __m128i {
        if remain >= 16 {
            return unsafe { _mm_loadu_si128(ptr.cast()) };
        }
        let mut tmp = [0u8; 16];
        if remain > 0 {
            unsafe {
                core::ptr::copy_nonoverlapping(ptr, tmp.as_mut_ptr(), remain);
            }
        }
        unsafe { _mm_loadu_si128(tmp.as_ptr().cast()) }
    }

    #[inline]
    unsafe fn shuffle_group(tag: u8, data: *const u8, remain: usize) -> (__m128i, usize) {
        let len = DATA_LEN[tag as usize] as usize;
        let shuf = unsafe { _mm_loadu_si128(DECODE_SHUFFLE[tag as usize].as_ptr().cast()) };
        let loaded = unsafe { load16(data, remain) };
        (unsafe { _mm_shuffle_epi8(loaded, shuf) }, len)
    }

    /// Decode `n` unsorted values from a Stream VByte buffer into `dst`.
    ///
    /// # Safety
    /// Caller must ensure SSSE3 is available and `dst.len() == n`.
    #[target_feature(enable = "ssse3")]
    pub unsafe fn decode_ssse3(n: usize, buf: &[u8], dst: &mut [u32]) {
        let n_groups = n.div_ceil(4);
        let tags = &buf[..n_groups];
        let data = &buf[n_groups..];
        let mut dp = 0usize;
        let mut i = 0;
        let mut g = 0;
        while i + 4 <= n {
            let tag = tags[g];
            let (vals, len) = unsafe { shuffle_group(tag, data.as_ptr().add(dp), data.len() - dp) };
            unsafe {
                _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), vals);
            }
            dp += len;
            g += 1;
            i += 4;
        }
        if i < n {
            let tag = tags[g];
            let (vals, _) = unsafe { shuffle_group(tag, data.as_ptr().add(dp), data.len() - dp) };
            let mut tmp = [0u32; 4];
            unsafe {
                _mm_storeu_si128(tmp.as_mut_ptr().cast(), vals);
            }
            dst[i..n].copy_from_slice(&tmp[..n - i]);
        }
    }

    /// Decode `n` sorted (delta-coded) values into `dst`.
    ///
    /// # Safety
    /// Caller must ensure SSSE3 is available and `dst.len() == n`.
    #[target_feature(enable = "ssse3")]
    pub unsafe fn decode_sorted_ssse3(n: usize, buf: &[u8], dst: &mut [u32]) {
        let n_groups = n.div_ceil(4);
        let tags = &buf[..n_groups];
        let data = &buf[n_groups..];
        let mut dp = 0usize;

        let first_tag = tags[0];
        let (deltas, len) = unsafe { shuffle_group(first_tag, data.as_ptr(), data.len()) };
        let prefix = prefix4(deltas);
        let first_n = n.min(4);
        if first_n == 4 {
            let vals = _mm_add_epi32(prefix, _mm_setr_epi32(0, 1, 2, 3));
            unsafe {
                _mm_storeu_si128(dst.as_mut_ptr().cast(), vals);
            }
        } else {
            let vals = _mm_add_epi32(prefix, _mm_setr_epi32(0, 1, 2, 3));
            let mut tmp = [0u32; 4];
            unsafe {
                _mm_storeu_si128(tmp.as_mut_ptr().cast(), vals);
            }
            dst[..first_n].copy_from_slice(&tmp[..first_n]);
            return;
        }
        dp += len;
        let mut prev = unsafe { *dst.as_ptr().add(3) };
        let mut i = 4;
        let mut g = 1;
        let ones = _mm_setr_epi32(1, 2, 3, 4);
        while i + 4 <= n {
            let tag = tags[g];
            let (deltas, len) =
                unsafe { shuffle_group(tag, data.as_ptr().add(dp), data.len() - dp) };
            let prefix = prefix4(deltas);
            let vals = _mm_add_epi32(prefix, _mm_add_epi32(_mm_set1_epi32(prev as i32), ones));
            unsafe {
                _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), vals);
            }
            prev = unsafe { *dst.as_ptr().add(i + 3) };
            dp += len;
            g += 1;
            i += 4;
        }
        if i < n {
            let tag = tags[g];
            let (deltas, _) = unsafe { shuffle_group(tag, data.as_ptr().add(dp), data.len() - dp) };
            let prefix = prefix4(deltas);
            let vals = _mm_add_epi32(prefix, _mm_add_epi32(_mm_set1_epi32(prev as i32), ones));
            let mut tmp = [0u32; 4];
            unsafe {
                _mm_storeu_si128(tmp.as_mut_ptr().cast(), vals);
            }
            dst[i..n].copy_from_slice(&tmp[..n - i]);
        }
    }

    #[inline]
    #[target_feature(enable = "ssse3")]
    fn prefix4(d: __m128i) -> __m128i {
        // [d0, d0+d1, d0+d1+d2, d0+d1+d2+d3]
        let t = _mm_add_epi32(d, _mm_bslli_si128(d, 4));
        _mm_add_epi32(t, _mm_bslli_si128(t, 8))
    }
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
    }

    #[test]
    fn max_value() {
        roundtrip_unsorted(&[u32::MAX]);
        roundtrip_unsorted(&[0, u32::MAX, 1, 2]);
        roundtrip_sorted(&[0, 1, u32::MAX]);
        roundtrip_sorted(&[u32::MAX]);
    }

    #[test]
    fn tails_one_two_three() {
        for n in 1..=16 {
            let list: Vec<u32> = (0..n).map(|i| i * 1000 + 7).collect();
            roundtrip_unsorted(&list);
            roundtrip_sorted(&list);
        }
    }

    #[test]
    fn random_widths_and_tails() {
        let mut rng = rand::rng();
        for n in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 63, 64, 65, 1000] {
            let list: Vec<u32> = (0..n)
                .map(|_| match rng.random_range(0..5u32) {
                    0 => rng.random_range(0..256),
                    1 => rng.random_range(256..65536),
                    2 => rng.random_range(65536..1 << 24),
                    3 => rng.random::<u32>(),
                    _ => 0,
                })
                .collect();
            roundtrip_unsorted(&list);

            let mut sorted = list;
            sorted.sort_unstable();
            sorted.dedup();
            if !sorted.is_empty() {
                roundtrip_sorted(&sorted);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn ssse3_matches_scalar() {
        if !is_x86_feature_detected!("ssse3") {
            return;
        }
        let mut rng = rand::rng();
        for n in [
            0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 128, 129, 1000,
        ] {
            let list: Vec<u32> = (0..n).map(|_| rng.random::<u32>()).collect();
            let mut buf = Vec::new();
            encode(&list, &mut buf);

            let mut scalar = Vec::new();
            decode_values_scalar(n, &buf, &mut scalar);
            let mut simd = vec![0u32; n];
            if n > 0 {
                unsafe {
                    kernel::decode_ssse3(n, &buf, &mut simd);
                }
            }
            assert_eq!(simd, scalar, "unsorted n={n}");

            let mut sorted = list;
            sorted.sort_unstable();
            sorted.dedup();
            let sn = sorted.len();
            buf.clear();
            encode_sorted(&sorted, &mut buf);
            let mut scalar = Vec::new();
            if sn == 0 {
                continue;
            }
            decode_sorted_scalar(sn, &buf, &mut scalar);
            let mut simd = vec![0u32; sn];
            unsafe {
                kernel::decode_sorted_ssse3(sn, &buf, &mut simd);
            }
            assert_eq!(simd, scalar, "sorted n={sn}");
        }
    }
}
