//! Stream VByte (Lemire & Kurz): groups of four values, a 2-bit length tag
//! per value packed into a control byte, all control bytes first, then the
//! data bytes. A tail of `n % 4` values is a final partial group.
//!
//! Sorted lists use [`crate::delta`]. The scalar codec *is* the format; the
//! SSSE3 kernel behind it must agree bit-for-bit.

use crate::delta;

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

/// pshufb control: gather used bytes of four LE u32s into a contiguous stream.
#[cfg(target_arch = "x86_64")]
const ENCODE_SHUFFLE: [[u8; 16]; 256] = {
    let mut t = [[0x80u8; 16]; 256];
    let mut tag = 0;
    while tag < 256 {
        t[tag] = encode_shuffle_entry(tag as u8);
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

#[cfg(target_arch = "x86_64")]
const fn encode_shuffle_entry(tag: u8) -> [u8; 16] {
    let mut shuf = [0x80u8; 16];
    let mut dst = 0usize;
    let mut i = 0;
    while i < 4 {
        let len = ((tag >> (2 * i)) & 3) as usize + 1;
        let mut j = 0;
        while j < len {
            shuf[dst] = (i * 4 + j) as u8;
            dst += 1;
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

fn write_group_at(vals: &[u32], n: usize, data: &mut [u8]) -> (u8, usize) {
    let mut tag = 0u8;
    let mut len = 0;
    for (i, &v) in vals[..n].iter().enumerate() {
        let t = tag_of(v);
        tag |= t << (2 * i);
        let nbytes = t as usize + 1;
        data[len..len + nbytes].copy_from_slice(&v.to_le_bytes()[..nbytes]);
        len += nbytes;
    }
    (tag, len)
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
    if n == 0 {
        return;
    }
    let n_groups = n.div_ceil(4);
    let tag_at = out.len();
    // 16 extra bytes so the SSSE3 store of the last full group cannot run past the buffer.
    out.resize(tag_at + n_groups + n * 4 + 16, 0);
    let data_len = {
        let buf = &mut out[tag_at..];
        let (tags, data) = buf.split_at_mut(n_groups);
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("ssse3") {
                // Safety: SSSE3 was detected; `data` has 16 bytes of padding.
                unsafe { kernel::encode_ssse3(list, tags, data) }
            } else {
                encode_values_into(list, tags, data)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        encode_values_into(list, tags, data)
    };
    out.truncate(tag_at + n_groups + data_len);
}

fn encode_values_into(list: &[u32], tags: &mut [u8], data: &mut [u8]) -> usize {
    let n = list.len();
    let mut dp = 0;
    let mut i = 0;
    let mut g = 0;
    while i + 4 <= n {
        let (tag, len) = write_group_at(&list[i..], 4, &mut data[dp..]);
        tags[g] = tag;
        dp += len;
        g += 1;
        i += 4;
    }
    if i < n {
        let (tag, len) = write_group_at(&list[i..], n - i, &mut data[dp..]);
        tags[g] = tag;
        dp += len;
    }
    dp
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
    if n == 0 {
        return;
    }
    let n_groups = n.div_ceil(4);
    let tag_at = out.len();
    out.resize(tag_at + n_groups + n * 4 + 16, 0);
    let data_len = {
        let buf = &mut out[tag_at..];
        let (tags, data) = buf.split_at_mut(n_groups);
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("ssse3") {
                // Safety: SSSE3 was detected; `data` has 16 bytes of padding.
                unsafe { kernel::encode_sorted_ssse3(list, tags, data) }
            } else {
                encode_sorted_into(list, tags, data)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        encode_sorted_into(list, tags, data)
    };
    out.truncate(tag_at + n_groups + data_len);
}

fn encode_sorted_into(list: &[u32], tags: &mut [u8], data: &mut [u8]) -> usize {
    let mut dp = 0;
    let mut prev = None;
    let mut grp = [0u32; 4];
    let mut k = 0;
    let mut g = 0;
    for &v in list {
        grp[k] = delta::gap(prev, v);
        prev = Some(v);
        k += 1;
        if k == 4 {
            let (tag, len) = write_group_at(&grp, 4, &mut data[dp..]);
            tags[g] = tag;
            dp += len;
            g += 1;
            k = 0;
        }
    }
    if k > 0 {
        let (tag, len) = write_group_at(&grp, k, &mut data[dp..]);
        tags[g] = tag;
        dp += len;
    }
    dp
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
// The kernel *is* the unsafe boundary: every fn requires SSSE3 and in-bounds
// pointers. Wrappers detect the feature and pass a valid encoding.
#[allow(unsafe_op_in_unsafe_fn)]
mod kernel {
    use super::{DATA_LEN, DECODE_SHUFFLE, ENCODE_SHUFFLE, tag_of, write_group_at};

    use std::arch::x86_64::{
        __m128i, _mm_add_epi32, _mm_alignr_epi8, _mm_bslli_si128, _mm_loadu_si128, _mm_set1_epi32,
        _mm_setr_epi32, _mm_shuffle_epi8, _mm_storeu_si128, _mm_sub_epi32,
    };

    /// Load 16 data bytes, zero-padding if fewer than 16 remain.
    ///
    /// # Safety
    /// SSSE3; `ptr` valid for `remain` bytes.
    #[inline]
    #[target_feature(enable = "ssse3")]
    unsafe fn load16(ptr: *const u8, remain: usize) -> __m128i {
        if remain >= 16 {
            return _mm_loadu_si128(ptr.cast());
        }
        let mut tmp = [0u8; 16];
        if remain > 0 {
            core::ptr::copy_nonoverlapping(ptr, tmp.as_mut_ptr(), remain);
        }
        _mm_loadu_si128(tmp.as_ptr().cast())
    }

    #[inline]
    #[target_feature(enable = "ssse3")]
    fn prefix4(d: __m128i) -> __m128i {
        let t = _mm_add_epi32(d, _mm_bslli_si128(d, 4));
        _mm_add_epi32(t, _mm_bslli_si128(t, 8))
    }

    #[inline]
    #[target_feature(enable = "ssse3")]
    unsafe fn shuffle_at(
        tags: *const u8,
        data: *const u8,
        g: usize,
        dp: usize,
        remain: usize,
    ) -> (__m128i, usize) {
        let tag = *tags.add(g);
        let shuf = _mm_loadu_si128(DECODE_SHUFFLE[tag as usize].as_ptr().cast());
        let loaded = load16(data.add(dp), remain);
        (
            _mm_shuffle_epi8(loaded, shuf),
            DATA_LEN[tag as usize] as usize,
        )
    }

    #[inline]
    #[target_feature(enable = "ssse3")]
    unsafe fn tag4(v: __m128i) -> u8 {
        let mut tmp = [0u32; 4];
        _mm_storeu_si128(tmp.as_mut_ptr().cast(), v);
        tag_of(tmp[0]) | (tag_of(tmp[1]) << 2) | (tag_of(tmp[2]) << 4) | (tag_of(tmp[3]) << 6)
    }

    /// Encode `list` into pre-sized `tags` and `data` (`data` has 16 extra bytes).
    ///
    /// # Safety
    /// SSSE3; `tags.len() == list.len().div_ceil(4)`; `data` holds worst-case plus 16.
    #[target_feature(enable = "ssse3")]
    pub unsafe fn encode_ssse3(list: &[u32], tags: &mut [u8], data: &mut [u8]) -> usize {
        let n = list.len();
        let n_full = n / 4;
        let mut dp = 0usize;
        let src = list.as_ptr();
        for (g, tag_slot) in tags.iter_mut().enumerate().take(n_full) {
            let v = _mm_loadu_si128(src.add(g * 4).cast());
            let tag = tag4(v);
            *tag_slot = tag;
            let shuf = _mm_loadu_si128(ENCODE_SHUFFLE[tag as usize].as_ptr().cast());
            _mm_storeu_si128(data.as_mut_ptr().add(dp).cast(), _mm_shuffle_epi8(v, shuf));
            dp += DATA_LEN[tag as usize] as usize;
        }
        if !n.is_multiple_of(4) {
            let (tag, len) = write_group_at(&list[n_full * 4..], n % 4, &mut data[dp..]);
            tags[n_full] = tag;
            dp += len;
        }
        dp
    }

    /// Encode sorted `list` as `crate::delta` then Stream VByte.
    ///
    /// # Safety
    /// Same as [`encode_ssse3`].
    #[target_feature(enable = "ssse3")]
    pub unsafe fn encode_sorted_ssse3(list: &[u32], tags: &mut [u8], data: &mut [u8]) -> usize {
        let n = list.len();
        let n_full = n / 4;
        let mut dp = 0usize;
        let src = list.as_ptr();
        let mut prev = _mm_set1_epi32(0);
        let ones = _mm_set1_epi32(1);
        let first_sub = _mm_setr_epi32(0, 1, 1, 1);
        for (g, tag_slot) in tags.iter_mut().enumerate().take(n_full) {
            let v = _mm_loadu_si128(src.add(g * 4).cast());
            let shifted = _mm_alignr_epi8(v, prev, 12);
            let raw = _mm_sub_epi32(v, shifted);
            let deltas = if g == 0 {
                _mm_sub_epi32(raw, first_sub)
            } else {
                _mm_sub_epi32(raw, ones)
            };
            prev = v;
            let tag = tag4(deltas);
            *tag_slot = tag;
            let shuf = _mm_loadu_si128(ENCODE_SHUFFLE[tag as usize].as_ptr().cast());
            _mm_storeu_si128(
                data.as_mut_ptr().add(dp).cast(),
                _mm_shuffle_epi8(deltas, shuf),
            );
            dp += DATA_LEN[tag as usize] as usize;
        }
        if !n.is_multiple_of(4) {
            let rest = &list[n_full * 4..];
            let mut grp = [0u32; 4];
            let mut prev_v = if n_full == 0 {
                None
            } else {
                Some(list[n_full * 4 - 1])
            };
            for (i, &v) in rest.iter().enumerate() {
                grp[i] = crate::delta::gap(prev_v, v);
                prev_v = Some(v);
            }
            let (tag, len) = write_group_at(&grp, n % 4, &mut data[dp..]);
            tags[n_full] = tag;
            dp += len;
        }
        dp
    }

    /// Decode `n` unsorted values from a Stream VByte buffer into `dst`.
    ///
    /// # Safety
    /// SSSE3; `dst.len() == n`; `buf` is a valid encoding of `n` values.
    #[target_feature(enable = "ssse3")]
    pub unsafe fn decode_ssse3(n: usize, buf: &[u8], dst: &mut [u32]) {
        let n_groups = n.div_ceil(4);
        let tags = buf.as_ptr();
        let data = buf.as_ptr().add(n_groups);
        let data_len = buf.len() - n_groups;
        let n_full = n / 4;
        let out = dst.as_mut_ptr();
        let mut dp = 0usize;
        let mut g = 0usize;
        while g < n_full && dp + 16 <= data_len {
            let tag = *tags.add(g);
            let shuf = _mm_loadu_si128(DECODE_SHUFFLE[tag as usize].as_ptr().cast());
            let vals = _mm_shuffle_epi8(_mm_loadu_si128(data.add(dp).cast()), shuf);
            _mm_storeu_si128(out.add(g * 4).cast(), vals);
            dp += DATA_LEN[tag as usize] as usize;
            g += 1;
        }
        while g < n_full {
            let tag = *tags.add(g);
            let shuf = _mm_loadu_si128(DECODE_SHUFFLE[tag as usize].as_ptr().cast());
            let vals = _mm_shuffle_epi8(load16(data.add(dp), data_len - dp), shuf);
            _mm_storeu_si128(out.add(g * 4).cast(), vals);
            dp += DATA_LEN[tag as usize] as usize;
            g += 1;
        }
        if !n.is_multiple_of(4) {
            let tag = *tags.add(g);
            let shuf = _mm_loadu_si128(DECODE_SHUFFLE[tag as usize].as_ptr().cast());
            let vals = _mm_shuffle_epi8(load16(data.add(dp), data_len.saturating_sub(dp)), shuf);
            let mut tmp = [0u32; 4];
            _mm_storeu_si128(tmp.as_mut_ptr().cast(), vals);
            core::ptr::copy_nonoverlapping(tmp.as_ptr(), out.add(n_full * 4), n % 4);
        }
    }

    /// Decode `n` sorted (delta-coded) values into `dst`.
    ///
    /// # Safety
    /// Same as [`decode_ssse3`].
    #[target_feature(enable = "ssse3")]
    pub unsafe fn decode_sorted_ssse3(n: usize, buf: &[u8], dst: &mut [u32]) {
        let n_groups = n.div_ceil(4);
        let tags = buf.as_ptr();
        let data = buf.as_ptr().add(n_groups);
        let data_len = buf.len() - n_groups;
        let n_full = n / 4;
        let out = dst.as_mut_ptr();
        let mut dp = 0usize;
        let ones = _mm_setr_epi32(1, 2, 3, 4);
        let first_add = _mm_setr_epi32(0, 1, 2, 3);

        if n_full == 0 {
            let (deltas, _) = shuffle_at(tags, data, 0, 0, data_len);
            let vals = _mm_add_epi32(prefix4(deltas), first_add);
            let mut tmp = [0u32; 4];
            _mm_storeu_si128(tmp.as_mut_ptr().cast(), vals);
            core::ptr::copy_nonoverlapping(tmp.as_ptr(), out, n);
            return;
        }

        // First group: prefix + [0,1,2,3].
        let (deltas, len) = shuffle_at(tags, data, 0, 0, data_len);
        let vals = _mm_add_epi32(prefix4(deltas), first_add);
        _mm_storeu_si128(out.cast(), vals);
        dp += len;
        let mut g = 1usize;
        let mut prev = *out.add(3);
        while g < n_full && dp + 16 <= data_len {
            let tag = *tags.add(g);
            let shuf = _mm_loadu_si128(DECODE_SHUFFLE[tag as usize].as_ptr().cast());
            let deltas = _mm_shuffle_epi8(_mm_loadu_si128(data.add(dp).cast()), shuf);
            let vals = _mm_add_epi32(
                prefix4(deltas),
                _mm_add_epi32(_mm_set1_epi32(prev as i32), ones),
            );
            _mm_storeu_si128(out.add(g * 4).cast(), vals);
            prev = *out.add(g * 4 + 3);
            dp += DATA_LEN[tag as usize] as usize;
            g += 1;
        }
        while g < n_full {
            let (deltas, len) = shuffle_at(tags, data, g, dp, data_len - dp);
            let vals = _mm_add_epi32(
                prefix4(deltas),
                _mm_add_epi32(_mm_set1_epi32(prev as i32), ones),
            );
            _mm_storeu_si128(out.add(g * 4).cast(), vals);
            prev = *out.add(g * 4 + 3);
            dp += len;
            g += 1;
        }
        if !n.is_multiple_of(4) {
            let (deltas, _) = shuffle_at(tags, data, g, dp, data_len.saturating_sub(dp));
            let vals = _mm_add_epi32(
                prefix4(deltas),
                _mm_add_epi32(_mm_set1_epi32(prev as i32), ones),
            );
            let mut tmp = [0u32; 4];
            _mm_storeu_si128(tmp.as_mut_ptr().cast(), vals);
            core::ptr::copy_nonoverlapping(tmp.as_ptr(), out.add(n_full * 4), n % 4);
        }
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
