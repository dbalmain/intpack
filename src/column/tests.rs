use super::*;

const WIDTHS: [u32; 10] = [0, 1, 2, 7, 8, 31, 57, 58, 63, 64];

fn pack(width: u32, values: &[u64]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut writer = Writer::new(width);
    for &v in values {
        assert!(writer.push(&mut out, v).is_ok());
    }
    assert!(writer.finish(&mut out).is_ok());
    out
}

/// Values that exercise every bit of a `width`-bit field: 0, all ones,
/// alternating patterns, and a pseudo-random spread.
fn values(width: u32, count: usize) -> Vec<u64> {
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    (0..count)
        .map(|i| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let v = match i % 5 {
                0 => 0,
                1 => u64::MAX,
                2 => 0x5555_5555_5555_5555,
                3 => 0xAAAA_AAAA_AAAA_AAAA,
                _ => x,
            };
            v & mask(width)
        })
        .collect()
}

#[test]
fn a_pass_yields_every_row_once_in_order_around_run_edges() {
    for count in [0, 1, RUN - 1, RUN, RUN + 1, 2 * RUN, 2 * RUN + 3] {
        let mut pass = runs(count, |first, out| {
            assert!(first + out.len() <= count, "decoded past row {count}");
            for (row, value) in (first..).zip(out.iter_mut()) {
                *value = row as u64;
            }
        });
        assert_eq!(pass.size_hint(), (count, Some(count)));
        let first: Vec<u64> = pass.by_ref().take(count / 2).collect();
        assert_eq!(
            pass.size_hint(),
            (count - count / 2, Some(count - count / 2))
        );
        let all: Vec<u64> = first.into_iter().chain(pass).collect();
        assert_eq!(all, (0..count as u64).collect::<Vec<_>>(), "{count}");
    }
}

#[test]
fn widths_and_masks_at_the_edges() {
    assert_eq!(
        [0, 1, 2, 3, 255, 256, u64::MAX].map(width),
        [0, 1, 2, 2, 8, 9, 64]
    );
    assert_eq!(mask(0), 0);
    assert_eq!(mask(1), 1);
    assert_eq!(mask(63), u64::MAX >> 1);
    assert_eq!(mask(64), u64::MAX);
    for w in 0..=MAX_WIDTH {
        assert_eq!(width(mask(w)), w);
    }
}

#[test]
fn every_width_round_trips_at_every_count_up_to_130() {
    // 130 values cross a 64-bit word boundary at every width, and every
    // count checks the last value against the padding.
    for w in 0..=MAX_WIDTH {
        for count in 0..130 {
            let values = values(w, count);
            let bytes = pack(w, &values);
            assert_eq!(bytes.len() as u64, len(count as u32, w), "w {w} n {count}");
            assert!(bytes.ends_with(&[0; PAD]), "w {w} n {count}");
            let column = Column::new(&bytes, w);
            for (i, &v) in values.iter().enumerate() {
                assert_eq!(column.get(i), v, "w {w} n {count} i {i}");
            }
            // Runs from each multiple of 8: short, a block, and to the
            // column's end, whose loads reach into the padding.
            for first in (0..count).step_by(8) {
                for len in [0, 1, 7, 8, 9, 64, count - first] {
                    let end = (first + len).min(count);
                    let mut out = vec![0; end - first];
                    column.decode(first, &mut out);
                    assert_eq!(out, values[first..end], "w {w} n {count} {first}..{end}");
                }
            }
        }
    }
}

#[test]
fn all_ones_and_zero_at_the_named_edges() {
    for w in WIDTHS {
        for count in [1, 2, 7, 8, 9, 64, 65] {
            for fill in [0, mask(w)] {
                let bytes = pack(w, &vec![fill; count]);
                let column = Column::new(&bytes, w);
                assert!((0..count).all(|i| column.get(i) == fill), "w {w} n {count}");
            }
        }
    }
}

#[test]
fn a_value_does_not_leak_into_its_neighbours() {
    // One all-ones value among zeros reads back alone at every position:
    // the straddling byte for widths above 57 is masked, not or-ed in
    // wholesale.
    for w in WIDTHS {
        for hot in 0..10 {
            let values: Vec<u64> = (0..10)
                .map(|i| if i == hot { mask(w) } else { 0 })
                .collect();
            let bytes = pack(w, &values);
            let column = Column::new(&bytes, w);
            for (i, &v) in values.iter().enumerate() {
                assert_eq!(column.get(i), v, "w {w} hot {hot} i {i}");
            }
        }
    }
}

#[test]
fn width_zero_and_count_zero_are_only_padding() {
    assert_eq!(pack(0, &[0; 1000]), [0; PAD]);
    assert_eq!(pack(64, &[]), [0; PAD]);
    assert_eq!(Column::new(&[0; 8], 0).get(999_999), 0);
}

/// A blocked column of `values`: its bytes, values length and widest
/// block, as a sizing pass and the writer find them.
fn pack_blocked(values: &[u64]) -> (Vec<u8>, u64, u32) {
    let mut sizer = BlockSizer::default();
    values.iter().for_each(|&v| sizer.push(v));
    let (len, widest) = sizer.finish();
    let (mut table, mut packed) = (Vec::new(), Vec::new());
    let mut writer = BlockedWriter::default();
    for &v in values {
        assert!(writer.push(&mut table, &mut packed, v).is_ok());
    }
    assert!(writer.finish(&mut table, &mut packed).is_ok());
    table.extend(packed);
    (table, len, widest)
}

/// Blocks of every width from 0 to 64, each block's values spread over
/// its width above a base that differs per block; the last block short.
fn blocked_values(count: usize) -> Vec<u64> {
    (0..count)
        .map(|i| {
            let (block, row) = (i / BLOCK_ROWS, i % BLOCK_ROWS);
            let w = (block * 7 % 65) as u32;
            let spread = values(w, BLOCK_ROWS)[row];
            // Width 64 has no room for a base; the others sit above one.
            let base = if w == 64 { 0 } else { (block as u64) << 40 };
            base.wrapping_add(spread)
        })
        .collect()
}

#[test]
fn a_blocked_column_round_trips_at_every_block_width() {
    let widths: Vec<u32> = (0..65).map(|b| (b * 7 % 65) as u32).collect();
    assert!(widths.contains(&0) && widths.contains(&64));
    for count in [0, 1, 8, 127, 128, 129, 1000, 65 * BLOCK_ROWS + 3] {
        let values = blocked_values(count);
        let (bytes, len, widest) = pack_blocked(&values);
        assert_eq!(
            bytes.len() as u64,
            blocked_len(count as u32, len),
            "n {count}"
        );
        let column = Blocked::new(&bytes, count as u32);
        assert!(column.check(widest, len, false), "n {count}");
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(column.get(i), v, "n {count} i {i}");
        }
        // Runs from multiples of 8, some across a block boundary.
        for first in (0..count).step_by(8) {
            for len in [1, 8, 64, 200] {
                let end = (first + len).min(count);
                let mut out = vec![0; end - first];
                column.decode(first, &mut out);
                assert_eq!(out, values[first..end], "n {count} {first}..{end}");
            }
        }
    }
}

#[test]
fn a_blocked_column_of_equal_values_is_only_its_table_and_padding() {
    let (bytes, len, widest) = pack_blocked(&[5; 300]);
    assert_eq!((len, widest), (0, 0));
    assert_eq!(bytes.len() as u64, 3 * BLOCK_ENTRY + PAD as u64);
    let column = Blocked::new(&bytes, 300);
    assert!((0..300).all(|i| column.get(i) == 5));
}

#[test]
fn a_blocked_table_that_misplaces_or_widens_a_block_is_refused() {
    // Three blocks: widths 3, 9 and 3 (the last short), so the second
    // block starts at 48 bytes and the third at 192.
    let values: Vec<u64> = (0..300)
        .map(|i| (i % 8) as u64 * if i / BLOCK_ROWS == 1 { 64 } else { 1 })
        .collect();
    let (bytes, len, widest) = pack_blocked(&values);
    assert_eq!(widest, 9);
    assert!(Blocked::new(&bytes, 300).check(widest, len, false));
    let set = |k: usize, offset: u64, width: u64| {
        let mut bad = bytes.clone();
        let at = k * BLOCK_ENTRY as usize + 8;
        bad[at..at + 8].copy_from_slice(&(offset << 8 | width).to_le_bytes());
        bad
    };
    assert!(
        Blocked::new(&set(1, 48, 9), 300).check(widest, len, false),
        "unchanged"
    );
    for (what, bad) in [
        ("wider than the column", set(1, 48, 10)),
        ("wider than 64", set(2, 192, 200)),
        ("an offset past its predecessor's end", set(1, 49, 9)),
        ("an offset before it", set(1, 47, 9)),
        // Narrower keeps the next offset wrong: the third block no
        // longer starts where the second ends.
        ("narrower than written", set(1, 48, 8)),
    ] {
        assert!(!Blocked::new(&bad, 300).check(widest, len, false), "{what}");
    }
    // The blocks agree with each other but not with the values length,
    // or no block is as wide as the descriptor says.
    let column = Blocked::new(&bytes, 300);
    assert!(!column.check(widest, len + 1, false) && !column.check(widest, len - 1, false));
    assert!(!column.check(widest + 1, len, false));
}

/// A nullable blocked column of `values`, as [`pack_blocked`].
fn pack_nullable(values: &[Option<u64>]) -> (Vec<u8>, u64, u32) {
    let mut sizer = BlockSizer::default();
    for &v in values {
        match v {
            Some(v) => sizer.push(v),
            None => sizer.push_null(),
        }
    }
    let (len, widest) = sizer.finish();
    let (mut table, mut packed) = (Vec::new(), Vec::new());
    let mut writer = BlockedWriter::default();
    for &v in values {
        let pushed = match v {
            Some(v) => writer.push(&mut table, &mut packed, v),
            None => writer.push_null(&mut table, &mut packed),
        };
        assert!(pushed.is_ok());
    }
    assert!(writer.finish(&mut table, &mut packed).is_ok());
    table.extend(packed);
    (table, len, widest)
}

#[test]
fn a_nullable_blocked_column_round_trips_nones_in_every_kind_of_block() {
    // Per block: no none (plain width, no flag), a few nones among
    // values (one more value's worth), nothing but nones (width 0),
    // one value and nones (width 1), and a spread of 63 bits with a
    // none (width 64); then the blocked values' widths, every third a
    // none. The last block is short.
    let kinds = 5 + 65;
    let count = kinds * BLOCK_ROWS + 3;
    let spread = blocked_values(count);
    let values: Vec<Option<u64>> = (0..count)
        .map(|i| {
            let (block, row) = (i / BLOCK_ROWS, i % BLOCK_ROWS);
            let base = (block as u64) << 32;
            match block {
                0 => Some(base + row as u64),
                1 => (row % 9 != 4).then_some(base + row as u64),
                2 => None,
                3 => (row == 17).then_some(base),
                4 if row == 5 => None,
                4 => Some(mask(63) & values(63, BLOCK_ROWS)[row]),
                _ if row % 3 == 0 => None,
                // Width 64 cannot hold a none past its values; the
                // spread stays below all ones.
                _ => Some(spread[i].min(u64::MAX - 1)),
            }
        })
        .collect();
    let (bytes, len, widest) = pack_nullable(&values);
    assert_eq!(bytes.len() as u64, blocked_len(count as u32, len));
    let column = Blocked::new(&bytes, count as u32);
    assert!(column.check(widest, len, true));
    assert!(!column.check(widest, len, false), "a none's flag");
    let width = |k: usize| column.entry(k).2;
    assert_eq!((width(0), width(1), width(2), width(3)), (7, 8, 0, 1));
    assert_eq!(width(4), 64);
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(column.nullable(i), v, "i {i}");
    }
    assert!(column.nullable_values().eq(values.iter().copied()));
}

#[test]
fn a_nullable_blocked_column_of_nothing_but_nones_is_only_its_table() {
    let (bytes, len, widest) = pack_nullable(&[None; 300]);
    assert_eq!((len, widest), (0, 0));
    assert_eq!(bytes.len() as u64, 3 * BLOCK_ENTRY + PAD as u64);
    let column = Blocked::new(&bytes, 300);
    assert!(column.check(0, 0, true));
    assert!((0..300).all(|i| column.nullable(i).is_none()));
    assert!(column.nullable_values().eq([None; 300]));
}

#[test]
fn a_read_past_the_padded_end_panics_rather_than_reading_garbage() {
    let bytes = pack(64, &[u64::MAX]);
    let column = Column::new(&bytes, 64);
    assert_eq!(column.get(0), u64::MAX);
    assert!(std::panic::catch_unwind(|| column.get(2)).is_err());
}

#[test]
fn a_column_is_a_bit_string_at_one_width_then_padding() {
    // The writer, the owned array and `bits::Writer` agree byte for byte, so
    // a column's unpadded run is readable through `bits::Reader` too.
    for w in 0..=MAX_WIDTH {
        for count in [0, 1, 7, 8, 9, 63, 64, 65, 130] {
            let values = values(w, count);
            let mut expected = Vec::new();
            let mut bits = crate::bits::Writer::new(&mut expected);
            values.iter().for_each(|&v| bits.put(v, w));
            let mut unpadded = Vec::new();
            let mut writer = Writer::new(w);
            for &v in &values {
                assert!(writer.push(&mut unpadded, v).is_ok());
            }
            assert!(writer.finish_unpadded(&mut unpadded).is_ok());
            assert_eq!(unpadded, expected, "w {w} n {count}");
            expected.extend_from_slice(&[0; PAD]);
            assert_eq!(pack(w, &values), expected, "w {w} n {count}");
            let array = Array::new(values.iter().copied());
            assert_eq!(array.len(), count);
            // An array takes the width of its largest value, which is the
            // width's all ones from the second value on.
            assert_eq!(
                array.width(),
                width(values.iter().copied().max().unwrap_or(0))
            );
            if array.width() == w {
                assert_eq!(array.as_bytes(), expected, "w {w} n {count}");
            }
            for (i, &v) in values.iter().enumerate() {
                assert_eq!(array.get(i), v, "w {w} n {count} i {i}");
            }
            if w <= 57 {
                let reader = crate::bits::Reader::new(&unpadded);
                for (i, &v) in values.iter().enumerate() {
                    assert_eq!(reader.get(i * w as usize, w), v, "w {w} i {i}");
                }
            }
        }
    }
}

#[test]
fn an_empty_array_is_width_zero_and_refuses_every_row() {
    let array = Array::new(std::iter::empty::<u64>());
    assert!(array.is_empty());
    assert_eq!((array.width(), array.as_bytes()), (0, &[0; PAD][..]));
    assert!(std::panic::catch_unwind(|| array.get(0)).is_err());
    let one = Array::new([u64::MAX]);
    assert_eq!((one.width(), one.get(0)), (64, u64::MAX));
    // The array, not only its padded bytes, bounds a read.
    assert!(std::panic::catch_unwind(|| one.get(1)).is_err());
    assert!(one.heap_bytes() >= len(1, 64) as usize);
}

/// The format is persisted by callers, so these bytes are frozen: a change
/// to the bit order, the padding, `BLOCK_ROWS`, the table entry or the
/// nulls flag fails here, not in someone's stored data.
#[test]
fn golden_bytes_pin_the_stable_format() {
    let le = |v: u64| v.to_le_bytes();
    // A plain column: 1 to 5 at width 3, LSB first, then the padding.
    let mut expected = vec![0xD1, 0x58];
    expected.extend_from_slice(&[0; PAD]);
    assert_eq!(pack(3, &[1, 2, 3, 4, 5]), expected);
    // A blocked column of 130 rows: a full block alternating 5 and 6 (base
    // 5, width 1), then a short block of two 9s (base 9, width 0) whose
    // values start 16 bytes in.
    let values: Vec<u64> = (0..130)
        .map(|i| if i < 128 { 5 + i % 2 } else { 9 })
        .collect();
    let mut expected = [le(5), le(1), le(9), le(16 << 8)].concat();
    expected.extend_from_slice(&[0xAA; 16]);
    expected.extend_from_slice(&[0; PAD]);
    assert_eq!(pack_blocked(&values).0, expected);
    // A nullable block: 7, none, 8 is base 7 at width 2 with the 0x80 flag;
    // the none is all ones: raw 0, 3, 1.
    let mut expected = [le(7), le(0x80 | 2)].concat();
    expected.push(0b01_11_00);
    expected.extend_from_slice(&[0; PAD]);
    assert_eq!(pack_nullable(&[Some(7), None, Some(8)]).0, expected);
}

#[test]
fn a_nullable_block_spanning_all_of_u64_is_refused_not_wrapped() {
    // Regression: the none's width was `max - min + 1`, which wrapped to 0
    // in release, so `check` passed and every row read back as none.
    let write = |values: &[Option<u64>]| {
        let (mut table, mut packed) = (Vec::new(), Vec::new());
        let mut writer = BlockedWriter::default();
        for &v in values {
            let pushed = match v {
                Some(v) => writer.push(&mut table, &mut packed, v),
                None => writer.push_null(&mut table, &mut packed),
            };
            assert!(pushed.is_ok());
        }
        writer.finish(&mut table, &mut packed)
    };
    let refused = write(&[Some(0), Some(u64::MAX), None]);
    assert_eq!(
        refused.map_err(|e| e.kind()).err(),
        Some(std::io::ErrorKind::InvalidInput)
    );
    // One less is the widest span a nullable block holds: width 64.
    let widest = [Some(0), Some(u64::MAX - 1), None];
    assert!(write(&widest).is_ok());
    let (bytes, len, width) = pack_nullable(&widest);
    assert_eq!(width, 64);
    let column = Blocked::new(&bytes, 3);
    assert!(column.check(width, len, true));
    assert!(column.nullable_values().eq(widest));
    // Without a none, the full span is an ordinary width-64 block.
    let (bytes, len, width) = pack_blocked(&[0, u64::MAX]);
    assert!(Blocked::new(&bytes, 2).check(width, len, false));
}

#[test]
fn check_refuses_a_width_past_64() {
    // A table width field holds up to 127; `check` is the whole trust gate,
    // so it refuses a caller's claimed widest past `MAX_WIDTH` itself.
    // One block of 8 rows at width 65 is 65 bytes, so the table agrees with
    // itself and with the claimed values length: only the width is wrong.
    let mut bytes = [0u64.to_le_bytes(), 65u64.to_le_bytes()].concat();
    bytes.extend_from_slice(&[0; 65 + PAD]);
    let column = Blocked::new(&bytes, 8);
    assert!(!column.check(65, 65, false));
    // The same table at width 64 is trusted.
    bytes[8] = 64;
    assert!(Blocked::new(&bytes, 8).check(64, 64, false));
}
