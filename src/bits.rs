//! Bit-granular writer and reader over a byte buffer, little-endian within
//! each byte. Shared by the Elias-Fano codecs and any format with
//! sub-byte fields. Not a hot path in itself: block codecs write whole words.

/// Appends bit fields to a `Vec<u8>`, LSB first.
pub struct Writer<'a> {
    out: &'a mut Vec<u8>,
    /// Bits already used in the last byte of `out` (0 when `out` ends on a
    /// byte boundary).
    fill: u32,
}

impl<'a> Writer<'a> {
    /// Start writing at the end of `out`, on a byte boundary.
    pub fn new(out: &'a mut Vec<u8>) -> Self {
        Self { out, fill: 0 }
    }

    /// Append the low `width` bits of `value` (`width <= 64`).
    pub fn put(&mut self, value: u64, width: u32) {
        debug_assert!(width <= 64);
        debug_assert!(width == 64 || value >> width == 0);
        let mut remaining = width;
        let mut value = value;
        while remaining > 0 {
            if self.fill == 0 {
                self.out.push(0);
            }
            let room = 8 - self.fill;
            let take = remaining.min(room);
            let last = self.out.len() - 1;
            self.out[last] |= ((value & ((1u64 << take) - 1)) as u8) << self.fill;
            value >>= take;
            remaining -= take;
            self.fill = (self.fill + take) % 8;
        }
    }

    /// Append `count` zero bits followed by a one bit — a unary code.
    pub fn put_unary(&mut self, count: u32) {
        let mut zeros = count;
        while zeros >= 64 {
            self.put(0, 64);
            zeros -= 64;
        }
        self.put(1u64 << zeros, zeros + 1);
    }

    /// Bits written so far through this writer, including partial bytes.
    pub fn bit_len(&self) -> usize {
        self.out.len() * 8
            - if self.fill == 0 {
                0
            } else {
                (8 - self.fill) as usize
            }
    }

    /// Pad to the next byte boundary.
    pub fn align(&mut self) {
        self.fill = 0;
    }
}

/// Reads bit fields at arbitrary bit offsets of a byte slice. Reads past the
/// end yield zero bits, so callers may over-read into padding.
#[derive(Clone, Copy)]
pub struct Reader<'a> {
    buf: &'a [u8],
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf }
    }

    /// The `width` bits starting at bit `pos` (`width <= 57`, so the field
    /// fits in one unaligned 8-byte load).
    #[inline]
    pub fn get(&self, pos: usize, width: u32) -> u64 {
        debug_assert!(width <= 57);
        let byte = pos / 8;
        let shift = (pos % 8) as u32;
        let word = self.load(byte) >> shift;
        if width == 0 {
            0
        } else {
            word & ((1u64 << width) - 1)
        }
    }

    /// One bit.
    #[inline]
    pub fn bit(&self, pos: usize) -> bool {
        self.buf
            .get(pos / 8)
            .is_some_and(|b| b >> (pos % 8) & 1 == 1)
    }

    /// The 64-bit word starting at bit `pos` (unaligned), zero-padded.
    #[inline]
    pub fn word(&self, pos: usize) -> u64 {
        let byte = pos / 8;
        let shift = (pos % 8) as u32;
        let lo = self.load(byte) >> shift;
        if shift == 0 {
            lo
        } else {
            lo | self.load(byte + 8) << (64 - shift)
        }
    }

    /// Eight bytes from `byte`, zero-padded past the end.
    #[inline]
    fn load(&self, byte: usize) -> u64 {
        match self.buf.get(byte..byte + 8) {
            Some(bytes) => u64::from_le_bytes(bytes.try_into().unwrap_or([0; 8])),
            None => {
                let mut bytes = [0u8; 8];
                if let Some(tail) = self.buf.get(byte..) {
                    bytes[..tail.len()].copy_from_slice(tail);
                }
                u64::from_le_bytes(bytes)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_roundtrip_across_byte_boundaries() {
        let mut out = Vec::new();
        let mut w = Writer::new(&mut out);
        let fields = [
            (5u64, 3u32),
            (0x1ff, 9),
            (0, 1),
            (u64::MAX, 64),
            (7, 57),
            (1, 1),
        ];
        for &(v, width) in &fields {
            w.put(v, width);
        }
        let total = w.bit_len();
        assert_eq!(total, 3 + 9 + 1 + 64 + 57 + 1);
        let r = Reader::new(&out);
        let mut pos = 0;
        for &(v, width) in &fields {
            if width <= 57 {
                assert_eq!(
                    r.get(pos, width),
                    v & ((1u64 << width) - 1),
                    "field at {pos}"
                );
            } else {
                assert_eq!(r.word(pos), v);
            }
            pos += width as usize;
        }
    }

    #[test]
    fn unary_and_bits() {
        let mut out = Vec::new();
        let mut w = Writer::new(&mut out);
        w.put_unary(0);
        w.put_unary(3);
        w.put_unary(70);
        let r = Reader::new(&out);
        assert!(r.bit(0));
        assert!(!r.bit(1) && !r.bit(2) && !r.bit(3) && r.bit(4));
        assert!((5..75).all(|i| !r.bit(i)) && r.bit(75));
        assert!(!r.bit(10_000));
    }

    #[test]
    fn reads_past_end_are_zero() {
        let r = Reader::new(&[0xff, 0xff]);
        assert_eq!(r.get(8, 16), 0xff);
        assert_eq!(r.word(16), 0);
    }
}
