//! Compact zero-copy state for repeated queries through an opened list.

use crate::bits::Reader;

use super::{Cursor, Layout, select_one, word_at};

/// A list opened once for repeated `get` and cursor queries.
pub struct View<'a> {
    buf: &'a [u8],
    n: usize,
    universe: u32,
    l: u32,
    highs_bit: usize,
    first_highs_word: u64,
    s1_super_start: usize,
    s1_sub_start: usize,
}

impl<'a> View<'a> {
    pub fn new(universe: u32, n: usize, buf: &'a [u8]) -> Self {
        let layout = Layout::with_samples(universe, n, buf);
        Self {
            buf,
            n,
            universe,
            l: layout.l,
            highs_bit: layout.highs_bit,
            first_highs_word: word_at(buf, layout.highs_bit),
            s1_super_start: layout.s1_super_start,
            s1_sub_start: layout.s1_sub_start,
        }
    }

    #[inline]
    pub fn get(&self, i: usize) -> u32 {
        assert!(i < self.n, "elias_fano::get({i}) out of range");
        let rank = i + 1;
        let ones = self.first_highs_word.count_ones() as usize;
        let pos = if self.s1_super_start == 0 && rank <= ones {
            crate::bits::select_one(self.first_highs_word, i) as usize
        } else {
            select_one(
                self.buf,
                self.highs_bit,
                self.s1_super_start,
                self.s1_sub_start,
                i,
            )
        };
        let high = (pos - i) as u32;
        let low = Reader::new(self.buf).get(8 + i * self.l as usize, self.l) as u32;
        ((u64::from(high) << self.l) | u64::from(low)) as u32
    }

    pub fn cursor(&self) -> Cursor<'a> {
        Cursor::new(self.universe, self.n, self.buf)
    }
}
