//! LEB128 varint: 7 payload bits per byte, continuation in the high bit.
//!
//! Sorted lists use [`crate::delta`]: first value as-is, then `gap - 1`. The
//! cursor is a linear scan — the honest baseline a skip structure has to beat.

use crate::Cursor;
use crate::delta;

/// Append one unsigned LEB128 value.
#[inline]
pub fn put(mut v: u32, out: &mut Vec<u8>) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Read one unsigned LEB128 value at `*pos`, advancing it.
///
/// The obvious byte loop; bulk decode uses a one-byte fast path around this.
#[inline]
pub fn take(buf: &[u8], pos: &mut usize) -> u32 {
    let mut v = 0u32;
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

#[inline]
fn take_fast(buf: &[u8], pos: &mut usize) -> u32 {
    let b = buf[*pos];
    if b < 0x80 {
        *pos += 1;
        u32::from(b)
    } else {
        take(buf, pos)
    }
}

/// Encode an unsorted list.
pub fn encode(list: &[u32], out: &mut Vec<u8>) {
    out.reserve(list.len());
    for &v in list {
        if v < 0x80 {
            out.push(v as u8);
        } else {
            put(v, out);
        }
    }
}

/// Decode `n` unsorted values, appending to `out`.
pub fn decode(n: usize, buf: &[u8], out: &mut Vec<u32>) {
    let start = out.len();
    out.resize(start + n, 0);
    let mut pos = 0;
    for slot in &mut out[start..] {
        *slot = take_fast(buf, &mut pos);
    }
}

/// Encode a strictly increasing list as deltas then varints.
pub fn encode_sorted(list: &[u32], out: &mut Vec<u8>) {
    out.reserve(list.len());
    let mut prev = None;
    for &v in list {
        let g = delta::gap(prev, v);
        if g < 0x80 {
            out.push(g as u8);
        } else {
            put(g, out);
        }
        prev = Some(v);
    }
}

/// Decode `n` sorted values, appending to `out`.
pub fn decode_sorted(n: usize, buf: &[u8], out: &mut Vec<u32>) {
    let start = out.len();
    out.resize(start + n, 0);
    let dst = &mut out[start..];
    if n == 0 {
        return;
    }
    let mut pos = 0;
    let mut prev = take_fast(buf, &mut pos);
    dst[0] = prev;
    for slot in &mut dst[1..] {
        let v = prev + take_fast(buf, &mut pos) + 1;
        *slot = v;
        prev = v;
    }
}

/// Linear-scan cursor over an encoded sorted list.
pub struct SortedCursor<'a> {
    buf: &'a [u8],
    pos: usize,
    remaining: usize,
    /// `None` before the first element.
    current: Option<u32>,
}

impl<'a> SortedCursor<'a> {
    pub fn new(n: usize, buf: &'a [u8]) -> Self {
        Self {
            buf,
            pos: 0,
            remaining: n,
            current: None,
        }
    }
}

impl Cursor for SortedCursor<'_> {
    fn next_geq(&mut self, target: u32) -> Option<u32> {
        if let Some(c) = self.current
            && c >= target
        {
            return Some(c);
        }
        while let Some(v) = self.next() {
            if v >= target {
                return Some(v);
            }
        }
        None
    }

    fn next(&mut self) -> Option<u32> {
        if self.remaining == 0 {
            self.current = None;
            return None;
        }
        self.remaining -= 1;
        // Not `take_fast`: on a one-element-at-a-time path its extra branch
        // costs more than it saves (cursor seek measured 1.6× slower with it).
        let gap = take(self.buf, &mut self.pos);
        let v = match self.current {
            Some(prev) => prev + gap + 1,
            None => gap,
        };
        self.current = Some(v);
        Some(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cursor;

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
        roundtrip_unsorted(&[0, u32::MAX, 1]);
        roundtrip_sorted(&[0, 1, u32::MAX]);
        roundtrip_sorted(&[u32::MAX]);
    }

    #[test]
    fn put_take_one_to_five_bytes() {
        let cases = [0, 127, 128, 0x3fff, 0x4000, 0x1fffff, 0x200000, u32::MAX];
        for v in cases {
            let mut buf = Vec::new();
            put(v, &mut buf);
            let mut pos = 0;
            assert_eq!(take(&buf, &mut pos), v);
            assert_eq!(pos, buf.len());
        }
    }

    #[test]
    fn cursor_next_and_next_geq() {
        let list: Vec<u32> = vec![1, 3, 10, 11, 100];
        let mut buf = Vec::new();
        encode_sorted(&list, &mut buf);

        let mut cur = SortedCursor::new(list.len(), &buf);
        for &v in &list {
            assert_eq!(cur.next(), Some(v));
        }
        assert_eq!(cur.next(), None);

        let mut cur = SortedCursor::new(list.len(), &buf);
        assert_eq!(cur.next_geq(0), Some(1));
        assert_eq!(cur.next_geq(1), Some(1), "stay on current");
        assert_eq!(cur.next_geq(2), Some(3));
        assert_eq!(cur.next_geq(3), Some(3));
        assert_eq!(cur.next_geq(11), Some(11));
        assert_eq!(cur.next_geq(12), Some(100));
        assert_eq!(cur.next_geq(100), Some(100));
        assert_eq!(cur.next_geq(101), None);
        assert_eq!(cur.next_geq(u32::MAX), None);
    }

    #[test]
    fn cursor_empty() {
        let mut cur = SortedCursor::new(0, &[]);
        assert_eq!(cur.next(), None);
        assert_eq!(cur.next_geq(0), None);
    }

    #[test]
    fn cursor_singleton() {
        let mut buf = Vec::new();
        encode_sorted(&[42], &mut buf);
        let mut cur = SortedCursor::new(1, &buf);
        assert_eq!(cur.next_geq(42), Some(42));
        assert_eq!(cur.next_geq(42), Some(42));
        assert_eq!(cur.next(), None);
    }
}
