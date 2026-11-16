//! Fixed-width columns: `count` values of `width` bits each (0 to 64), value
//! `i` starting at bit `i * width` of a little-endian, LSB-first bit string —
//! the bit order of [`bits`], so a column is exactly what a
//! [`bits::Writer`] putting every value at one width
//! writes, plus padding. Where the codecs elsewhere in the crate decode a
//! list from its start, a column is read **by row**, in O(1), in place.
//!
//! A read is one unaligned 8-byte load, a shift and a mask, plus one more
//! byte when the value straddles the 8-byte window (`shift + width > 64`,
//! which only widths 58 to 63 can: width 64 always has shift 0). Every column
//! ends in [`PAD`] bytes of padding, so the load never runs off the end: the
//! writer writes zeros, readers never depend on their value, and [`len`] is
//! the exact length, padding included, which a caller reading untrusted bytes
//! should hold the column to before reading. Width 0 is legal: every value is
//! 0, and the column is only its padding.
//!
//! The values are raw. A frame of reference, a dictionary or a sentinel is
//! the caller's business, except in a [`Blocked`] column, which carries a
//! frame per block of [`BLOCK_ROWS`] rows. [`Array`] is an owned column for
//! values held in memory.
//!
//! Writers stream to any [`std::io::Write`], holding at most one word, so a
//! column never needs to be held whole to be written.

use std::io::{self, Write};

use crate::bits;

mod blocked;
pub use blocked::{
    BLOCK_ENTRY, BLOCK_ROWS, BlockSizer, Blocked, BlockedWriter, blocked_len, blocks,
};

/// Bytes of padding after a column's last value, written as zeros.
pub const PAD: usize = 8;

/// The widest a column can be.
pub const MAX_WIDTH: u32 = 64;

/// The bits needed to hold every value up to `max`: 0 for 0.
pub fn width(max: u64) -> u32 {
    u64::BITS - max.leading_zeros()
}

/// The largest value `width` bits hold: all ones. `width` is at most
/// [`MAX_WIDTH`].
#[inline]
pub fn mask(width: u32) -> u64 {
    u64::MAX.checked_shr(u64::BITS - width).unwrap_or(0)
}

/// A column's exact length in bytes, padding included. Cannot overflow for
/// any `u32` count and width up to [`MAX_WIDTH`].
pub fn len(count: u32, width: u32) -> u64 {
    (u64::from(count) * u64::from(width)).div_ceil(8) + PAD as u64
}

/// A column's bytes, read in place.
#[derive(Clone, Copy)]
pub struct Column<'a> {
    bytes: &'a [u8],
    width: u32,
    mask: u64,
}

impl<'a> Column<'a> {
    /// `bytes` must be at least [`len`] of the column's count and `width`,
    /// and `width` at most [`MAX_WIDTH`]; a read the bytes cannot cover
    /// panics, as indexing a slice does.
    #[inline]
    pub fn new(bytes: &'a [u8], width: u32) -> Self {
        Self {
            bytes,
            width,
            mask: mask(width),
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    /// Value `i`. Panics past the column's end, as indexing a slice does.
    #[inline]
    pub fn get(&self, i: usize) -> u64 {
        let bit = i * self.width as usize;
        let (at, shift) = (bit / 8, (bit % 8) as u32);
        let mut word = [0; 8];
        word.copy_from_slice(&self.bytes[at..at + 8]);
        let mut value = u64::from_le_bytes(word) >> shift;
        if shift + self.width > u64::BITS {
            value |= u64::from(self.bytes[at + 8]) << (u64::BITS - shift);
        }
        value & self.mask
    }

    /// Values `first..first + out.len()` into `out`, for a pass over the
    /// column; `first` is a multiple of 8, and the values are inside the
    /// column. Through [`Column::get`] each value re-derives its byte and
    /// shift and handles the straddling byte; here the run starts on a byte,
    /// and only widths above 57 take a second loop. Measured on a 10M-row
    /// column validated in passes: 0.18 s of user time through `get`, 0.14 s
    /// through this; a bit-buffer iterator was no faster than `get`.
    pub fn decode(&self, first: usize, out: &mut [u64]) {
        debug_assert!(first.is_multiple_of(8));
        let width = self.width as usize;
        let start = first / 8 * width;
        let end = start + (out.len() * width).div_ceil(8) + PAD;
        let bytes = &self.bytes[start..end];
        for (j, value) in out.iter_mut().enumerate() {
            let bit = j * width;
            let mut word = [0; 8];
            word.copy_from_slice(&bytes[bit / 8..bit / 8 + 8]);
            *value = u64::from_le_bytes(word) >> (bit % 8) & self.mask;
        }
        if width > 57 {
            // Only widths 58 to 63 can straddle the 8-byte window (64 has shift
            // 0).
            for (j, value) in out.iter_mut().enumerate() {
                let (bit, shift) = (j * width, (j * width % 8) as u32);
                if shift as usize + width > 64 {
                    *value |= u64::from(bytes[bit / 8 + 8]) << (u64::BITS - shift) & self.mask;
                }
            }
        }
    }
}

/// Packs values as they are pushed, so a column streams to the output
/// without being held.
pub struct Writer {
    width: u32,
    mask: u64,
    /// Bits not yet written, low first; fewer than 64 between pushes.
    pending: u64,
    filled: u32,
}

impl Writer {
    /// `width` is at most [`MAX_WIDTH`].
    pub fn new(width: u32) -> Self {
        debug_assert!(width <= MAX_WIDTH);
        Self {
            width,
            mask: mask(width),
            pending: 0,
            filled: 0,
        }
    }

    /// Appends `value`, which must fit the width; extra high bits are
    /// dropped (and fail a debug assertion).
    pub fn push(&mut self, out: &mut impl Write, value: u64) -> io::Result<()> {
        debug_assert!(value <= self.mask, "{value} does not fit {}", self.width);
        let value = value & self.mask;
        self.pending |= value << self.filled;
        self.filled += self.width;
        if self.filled >= u64::BITS {
            out.write_all(&self.pending.to_le_bytes())?;
            self.filled -= u64::BITS;
            // The bits that did not fit: none when the word was empty.
            self.pending = value.checked_shr(self.width - self.filled).unwrap_or(0);
        }
        Ok(())
    }

    /// Writes the last partial word and the [`PAD`] bytes: a column.
    pub fn finish(self, out: &mut impl Write) -> io::Result<()> {
        self.finish_unpadded(out)?;
        out.write_all(&[0; PAD])
    }

    /// Writes the last partial word, without padding: a bit string of
    /// exactly `ceil(count * width / 8)` bytes, for a run that something else
    /// follows (a block of a [`Blocked`] column) or one read through a reader
    /// that needs no padding ([`bits::Reader`]).
    pub fn finish_unpadded(self, out: &mut impl Write) -> io::Result<()> {
        let tail = self.filled.div_ceil(8) as usize;
        out.write_all(&self.pending.to_le_bytes()[..tail])
    }
}

/// Values in one decoded run of a pass over a column.
pub const RUN: usize = 64;

/// Rows `0..count` of a column in order, `decode`d [`RUN`] at a time: a pass
/// over the whole column, for a validation or a scan. `decode(first, out)`
/// fills `out` with rows from `first`, a multiple of [`RUN`], as
/// [`Column::decode`] and [`Blocked::decode`] do.
pub fn runs(count: usize, decode: impl Fn(usize, &mut [u64])) -> impl Iterator<Item = u64> {
    Runs {
        decode,
        count,
        next: 0,
        at: 0,
        len: 0,
        run: [0; RUN],
    }
}

/// The pass [`runs`] returns: the run decoded last, and where the next
/// starts.
struct Runs<F> {
    decode: F,
    count: usize,
    /// The first row of the run after this one.
    next: usize,
    /// The next value's index in `run`, and how many it holds.
    at: usize,
    len: usize,
    run: [u64; RUN],
}

impl<F: Fn(usize, &mut [u64])> Iterator for Runs<F> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        if self.at == self.len {
            if self.next == self.count {
                return None;
            }
            self.len = RUN.min(self.count - self.next);
            (self.decode)(self.next, &mut self.run[..self.len]);
            self.next += self.len;
            self.at = 0;
        }
        let value = self.run[self.at];
        self.at += 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.count - self.next + self.len - self.at;
        (left, Some(left))
    }
}

/// An owned column, at the width of its largest value.
pub struct Array {
    width: u32,
    len: usize,
    bytes: Vec<u8>,
}

impl Array {
    /// Packs `values` at the width of the largest (0 when there are none).
    /// The iterator is walked twice: once for the width, once to pack.
    pub fn new<I>(values: I) -> Self
    where
        I: IntoIterator<Item = u64>,
        I::IntoIter: ExactSizeIterator + Clone,
    {
        let values = values.into_iter();
        let len = values.len();
        let width = width(values.clone().max().unwrap_or(0));
        let mut bytes = Vec::with_capacity((len * width as usize).div_ceil(8) + PAD);
        let mut writer = bits::Writer::new(&mut bytes);
        for value in values {
            writer.put(value, width);
        }
        bytes.extend_from_slice(&[0; PAD]);
        Self { width, len, bytes }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    /// Value `i`. Panics at or past [`Array::len`].
    #[inline]
    pub fn get(&self, i: usize) -> u64 {
        assert!(i < self.len, "column row {i} of {}", self.len);
        self.column().get(i)
    }

    /// The array as a borrowed [`Column`].
    #[inline]
    pub fn column(&self) -> Column<'_> {
        Column::new(&self.bytes, self.width)
    }

    /// The heap bytes the array holds, capacity included.
    pub fn heap_bytes(&self) -> usize {
        self.bytes.capacity()
    }

    /// The packed bytes, padding included.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[cfg(test)]
mod tests;
