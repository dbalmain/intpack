//! [`Blocked`] columns: a frame of reference per block of [`BLOCK_ROWS`]
//! rows, each block packed at its own width, with the table that locates
//! them; and the writer and sizer that build one. A column may be
//! **nullable**: a row may hold no value, which costs nothing in a block
//! without one.

use super::*;

/// Rows per block of a [`Blocked`] column. A power of two and a multiple of
/// 64, so a validation run ([`Column::decode`]'s) never spans two blocks and
/// every full block ends on a byte. Measured on a 10M-row desktop file
/// catalog's columns, from 64 to 4096: 128 is within 2% of the best size for
/// byte offsets, file sizes and times, and 256 or more for link counts,
/// whose blocks are nearly all width 0; 128 was the smallest total over
/// those columns.
pub const BLOCK_ROWS: usize = 128;

/// Bytes per block in a blocked column's table: the block's base, then its
/// values' byte offset shifted left 8 with its width in the low byte.
pub const BLOCK_ENTRY: u64 = 16;

/// Set beside the width in a nullable column's table entry when the block
/// holds a row with no value: the width's all-ones value then means none,
/// and a block of nothing else is width 0. A block with every value present
/// keeps its plain width.
const NULLS: u64 = 0x80;

/// The blocks a blocked column of `count` rows has.
pub fn blocks(count: u32) -> u64 {
    u64::from(count).div_ceil(BLOCK_ROWS as u64)
}

/// A blocked column's exact length in bytes: its block table, `values` bytes
/// of packed blocks, and the padding. Saturates rather than overflow, since
/// `values` comes from the file: no section is that long.
pub fn blocked_len(count: u32, values: u64) -> u64 {
    (blocks(count) * BLOCK_ENTRY)
        .saturating_add(values)
        .saturating_add(PAD as u64)
}

/// The bytes of block `k`'s values for its `rows` rows at `width` bits.
fn block_bytes(rows: usize, width: u32) -> u64 {
    (rows as u64 * u64::from(width)).div_ceil(8)
}

/// A column in blocks of [`BLOCK_ROWS`] rows, each with its own frame of
/// reference: value `i` is its block's base plus a packed value at the
/// block's own width. A column whose neighbouring rows are close (offsets,
/// which only grow; counts that are nearly always 1) is narrower per block
/// than over the whole column. A read is two table loads more than
/// [`Column`]'s, and O(1).
///
/// A nullable blocked column frames each block by its real values alone,
/// and marks a block that holds a none with `NULLS`; those real values must
/// span less than `u64::MAX` (see [`BlockedWriter::push_null`]).
///
/// ```text
/// table    per block: base u64, (offset << 8 | nulls | width) u64
///          (BLOCK_ENTRY bytes; nulls is 0x80, width the low 7 bits)
/// values   each block's values, packed LSB first from a byte boundary
/// padding  PAD bytes
/// ```
#[derive(Clone, Copy)]
pub struct Blocked<'a> {
    table: &'a [u8],
    values: &'a [u8],
    count: usize,
}

impl<'a> Blocked<'a> {
    /// `bytes` must hold at least the table of a `count`-row column. Reads
    /// trust the table: a caller reading untrusted bytes holds them to
    /// [`blocked_len`] and calls [`Blocked::check`] before any read.
    pub fn new(bytes: &'a [u8], count: u32) -> Self {
        let (table, values) = bytes.split_at((blocks(count) * BLOCK_ENTRY) as usize);
        Self {
            table,
            values,
            count: count as usize,
        }
    }

    /// Block `k`'s base, values offset and width.
    pub(super) fn entry(&self, k: usize) -> (u64, usize, u32) {
        let (base, packed) = self.words(k);
        (base, (packed >> 8) as usize, (packed & 0x7F) as u32)
    }

    /// Block `k`'s two table words.
    fn words(&self, k: usize) -> (u64, u64) {
        let at = k * BLOCK_ENTRY as usize;
        let word = |at: usize| {
            let mut b = [0; 8];
            b.copy_from_slice(&self.table[at..at + 8]);
            u64::from_le_bytes(b)
        };
        (word(at), word(at + 8))
    }

    /// Whether every entry is one a read can trust: each offset exactly
    /// where the blocks before it end, the last block ending at `values`
    /// bytes (the values length the writer's [`BlockSizer`] found), and the
    /// widest block exactly `max_width`, which is at most [`MAX_WIDTH`]: so
    /// every width is too, and the caller's record of both is exact. Only a
    /// `nullable` column may mark a block as holding a none.
    pub fn check(&self, max_width: u32, values: u64, nullable: bool) -> bool {
        if max_width > MAX_WIDTH {
            return false;
        }
        let (mut end, mut widest) = (0, 0);
        for k in 0..self.table.len() / BLOCK_ENTRY as usize {
            let (_, offset, width) = self.entry(k);
            if offset as u64 != end || (!nullable && self.words(k).1 & NULLS != 0) {
                return false;
            }
            end += block_bytes(BLOCK_ROWS.min(self.count - k * BLOCK_ROWS), width);
            widest = widest.max(width);
        }
        end == values && widest == max_width
    }

    /// Value `i`, below the column's count (a debug assertion); past the
    /// table, it panics as indexing a slice does.
    pub fn get(&self, i: usize) -> u64 {
        debug_assert!(i < self.count, "row {i} of {}", self.count);
        let (base, offset, width) = self.entry(i / BLOCK_ROWS);
        base.wrapping_add(Column::new(&self.values[offset..], width).get(i % BLOCK_ROWS))
    }

    /// Value `i` of a nullable column, or `None` where it holds none.
    pub fn nullable(&self, i: usize) -> Option<u64> {
        debug_assert!(i < self.count, "row {i} of {}", self.count);
        let (base, packed) = self.words(i / BLOCK_ROWS);
        let (offset, width) = ((packed >> 8) as usize, (packed & 0x7F) as u32);
        let raw = Column::new(&self.values[offset..], width).get(i % BLOCK_ROWS);
        (packed & NULLS == 0 || raw != mask(width)).then(|| base.wrapping_add(raw))
    }

    /// Every value of a nullable column in order, `None` where it holds none:
    /// a pass over the column that decodes a block at a time, where
    /// [`Blocked::nullable`] re-derives each row's block and position.
    pub fn nullable_values(&self) -> impl Iterator<Item = Option<u64>> + '_ {
        (0..self.count.div_ceil(BLOCK_ROWS)).flat_map(move |k| {
            let (base, packed) = self.words(k);
            let (offset, width) = ((packed >> 8) as usize, (packed & 0x7F) as u32);
            let rows = BLOCK_ROWS.min(self.count - k * BLOCK_ROWS);
            let mut raw = [0; BLOCK_ROWS];
            Column::new(&self.values[offset..], width).decode(0, &mut raw[..rows]);
            let none = (packed & NULLS != 0).then(|| mask(width));
            raw.into_iter()
                .take(rows)
                .map(move |raw| (none != Some(raw)).then(|| base.wrapping_add(raw)))
        })
    }

    /// Values `first..first + out.len()` into `out`, a run of a pass over the
    /// column; `first` is a multiple of 8. See [`Column::decode`].
    pub fn decode(&self, first: usize, out: &mut [u64]) {
        // An empty run may start anywhere, past the end included.
        debug_assert!(
            out.is_empty() || first + out.len() <= self.count,
            "rows past {}",
            self.count
        );
        let mut done = 0;
        while done < out.len() {
            let row = first + done;
            let (base, offset, width) = self.entry(row / BLOCK_ROWS);
            let n = (BLOCK_ROWS - row % BLOCK_ROWS).min(out.len() - done);
            let run = &mut out[done..done + n];
            Column::new(&self.values[offset..], width).decode(row % BLOCK_ROWS, run);
            for value in run {
                *value = base.wrapping_add(*value);
            }
            done += n;
        }
    }
}

/// Finds a blocked column's values length and widest block from its values
/// in order, holding one block's range.
#[derive(Default)]
pub struct BlockSizer {
    range: Option<(u64, u64)>,
    nulls: bool,
    rows: usize,
    values: u64,
    width: u32,
}

impl BlockSizer {
    #[inline]
    pub fn push(&mut self, value: u64) {
        let (min, max) = self.range.unwrap_or((value, value));
        self.range = Some((min.min(value), max.max(value)));
        self.next();
    }

    /// A nullable column's none. A block holding one must have real values
    /// spanning less than `u64::MAX`, as [`BlockedWriter::push_null`] says;
    /// a block that does not is sized at [`MAX_WIDTH`], and the writer
    /// refuses it.
    #[inline]
    pub fn push_null(&mut self) {
        self.nulls = true;
        self.next();
    }

    #[inline]
    fn next(&mut self) {
        self.rows += 1;
        if self.rows == BLOCK_ROWS {
            self.close();
        }
    }

    fn close(&mut self) {
        if let Some((min, max)) = self.range.take() {
            let w = block_width(min, max, self.nulls).unwrap_or(MAX_WIDTH);
            self.values += block_bytes(self.rows, w);
            self.width = self.width.max(w);
        }
        (self.rows, self.nulls) = (0, false);
    }

    /// The bytes of packed values, and the widest block.
    pub fn finish(mut self) -> (u64, u32) {
        self.close();
        (self.values, self.width)
    }
}

/// A block's width for real values from `min` to `max`: one more value's
/// worth when the block holds a none, whose all-ones must be past them all.
/// `None` when that does not fit 64 bits: a block with a none whose values
/// span all of `u64`.
fn block_width(min: u64, max: u64, nulls: bool) -> Option<u32> {
    (max - min).checked_add(u64::from(nulls)).map(width)
}

/// Streams a blocked column: holds one block of values, then writes its
/// table entry to one output and its packed values to another, so neither
/// the table nor the values are ever held whole.
pub struct BlockedWriter {
    block: [u64; BLOCK_ROWS],
    /// Bit `i` set: row `i` of the block is none.
    nulls: u128,
    rows: usize,
    offset: u64,
}

impl Default for BlockedWriter {
    fn default() -> Self {
        Self {
            block: [0; BLOCK_ROWS],
            nulls: 0,
            rows: 0,
            offset: 0,
        }
    }
}

impl BlockedWriter {
    pub fn push(
        &mut self,
        table: &mut impl Write,
        values: &mut impl Write,
        value: u64,
    ) -> io::Result<()> {
        self.block[self.rows] = value;
        self.next(table, values)
    }

    /// A nullable column's none. Its block's real values must span less
    /// than `u64::MAX`, so that all ones above the smallest is past them all
    /// (`Some(0)`, `Some(u64::MAX)` and a none cannot share a block): the
    /// write that closes such a block fails with
    /// [`io::ErrorKind::InvalidInput`], and the column is not usable.
    pub fn push_null(&mut self, table: &mut impl Write, values: &mut impl Write) -> io::Result<()> {
        self.nulls |= 1 << self.rows;
        self.next(table, values)
    }

    fn next(&mut self, table: &mut impl Write, values: &mut impl Write) -> io::Result<()> {
        self.rows += 1;
        if self.rows == BLOCK_ROWS {
            self.close(table, values)?;
        }
        Ok(())
    }

    fn close(&mut self, table: &mut impl Write, values: &mut impl Write) -> io::Result<()> {
        if self.rows == 0 {
            return Ok(());
        }
        let nulls = self.nulls;
        let is_null = |i: usize| nulls >> i & 1 == 1;
        let block = &self.block[..self.rows];
        let real = || (0..block.len()).filter(|&i| !is_null(i)).map(|i| block[i]);
        // A block of nothing but nones has base 0 and width 0.
        let (min, w) = match (real().min(), real().max()) {
            (Some(min), Some(max)) => {
                let w = block_width(min, max, nulls != 0).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "a nullable block's values span all of u64",
                    )
                })?;
                (min, w)
            }
            _ => (0, 0),
        };
        let flag = if nulls == 0 { 0 } else { NULLS };
        table.write_all(&min.to_le_bytes())?;
        table.write_all(&(self.offset << 8 | flag | u64::from(w)).to_le_bytes())?;
        let mut writer = Writer::new(w);
        for (i, &value) in block.iter().enumerate() {
            let raw = if is_null(i) { mask(w) } else { value - min };
            writer.push(values, raw)?;
        }
        writer.finish_unpadded(values)?;
        self.offset += block_bytes(self.rows, w);
        (self.rows, self.nulls) = (0, 0);
        Ok(())
    }

    /// Writes the last partial block and the padding.
    pub fn finish(mut self, table: &mut impl Write, values: &mut impl Write) -> io::Result<()> {
        self.close(table, values)?;
        values.write_all(&[0; PAD])
    }
}
