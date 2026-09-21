//! Bitpacking of one 128-`u32` block at a fixed bit width — the primitive
//! under [`pfor128`](crate::pfor128) and the crate's worked example of the
//! kernel rule: [`scalar`] is the specification, written in the plainest
//! form; [`simd`] does the same work four lanes at a time and must produce
//! identical bytes (the tests here hold it to that for every width). The
//! entry points below are the one place that knows there are two.
//!
//! # Layout
//!
//! A block is `width * 16` bytes, `width` in `0..=32`. It is four interleaved
//! lanes: element `i` belongs to lane `i % 4`, and each lane is an independent
//! LSB-first bit stream of its 32 elements. The lanes' 32-bit words are
//! interleaved, so the packed bytes are `width` groups of four words (one per
//! lane) — exactly one 128-bit vector per group, which is why a vector
//! kernel over `[u32; 4]` groups and a scalar loop over the same groups do
//! the same thing. Width 0 packs to nothing. Same layout as the `bitpacking`
//! crate's `BitPacker4x`.
//!
//! # Sorted blocks
//!
//! Sorted lists store gaps (see [`delta`](crate::delta)). [`width_sorted`],
//! [`pack_sorted`] and [`unpack_sorted`] fuse the delta coding with the
//! packing, so the kernel never writes the gaps out; their specification is
//! literally the gaps into a scratch block, then the unsorted operation (or,
//! unpacking, [`unpack`] then [`delta::decode`](crate::delta::decode)).

/// The specification is compiled when it is the implementation or the test
/// oracle.
#[cfg(any(test, not(all(target_arch = "x86_64", target_feature = "sse2"))))]
mod scalar;
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
mod simd;

/// Elements per block.
pub const BLOCK: usize = 128;

/// Bits needed to hold every value of `block`: `0` for an all-zero block.
pub fn width(block: &[u32; BLOCK]) -> u32 {
    32 - block.iter().fold(0, |acc, &v| acc | v).leading_zeros()
}

/// Bytes a block packs to at `width`.
pub fn packed_len(width: u32) -> usize {
    width as usize * 16
}

/// The low `width` bits set.
const fn mask(width: u32) -> u32 {
    ((1u64 << width) - 1) as u32
}

/// Instantiate a const-generic kernel for a runtime width: the one sanctioned
/// 33-arm match, so shift amounts are constants inside the kernel.
///
/// SAFETY: the kernels are `#[target_feature(enable = "sse2")]`, and this
/// macro is only compiled where `cfg(target_feature = "sse2")` holds, so the
/// feature is present without runtime detection.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
macro_rules! with_width {
    ($width:expr, $kernel:ident :: $f:ident $args:tt) => {
        with_width!(@ $width, $kernel::$f, $args,
            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
            17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32)
    };
    (@ $width:expr, $kernel:ident :: $f:ident, $args:tt, $($w:literal)*) => {
        match $width {
            $($w => unsafe { $kernel::$f::<$w> $args },)*
            _ => unreachable!("width checked by the caller"),
        }
    };
}

/// Pack `block` at `width` into `out`, which must be exactly
/// `packed_len(width)` bytes. Every value must fit in `width` bits (see
/// [`width`]); a wider value corrupts its neighbours.
pub fn pack(width: u32, block: &[u32; BLOCK], out: &mut [u8]) {
    check(width, out.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    with_width!(width, simd::pack(block, out));
    #[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
    scalar::pack(width, block, out);
}

/// Inverse of [`pack`]: `packed` must be exactly `packed_len(width)` bytes.
pub fn unpack(width: u32, packed: &[u8], out: &mut [u32; BLOCK]) {
    check(width, packed.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    with_width!(width, simd::unpack(packed, out));
    #[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
    scalar::unpack(width, packed, out);
}

/// [`width`] of the gaps of a strictly increasing `block`, `base` being the
/// element before it (`None` at the head of a list), per the crate's delta
/// convention.
pub fn width_sorted(base: Option<u32>, block: &[u32; BLOCK]) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    // SAFETY: as `with_width!`.
    unsafe {
        simd::width_sorted(base, block)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
    scalar::width_sorted(base, block)
}

/// [`pack`] the gaps of a strictly increasing `block` at `width`, which must
/// be at least [`width_sorted`] of it.
pub fn pack_sorted(width: u32, base: Option<u32>, block: &[u32; BLOCK], out: &mut [u8]) {
    check(width, out.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    with_width!(width, simd::pack_sorted(base, block, out));
    #[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
    scalar::pack_sorted(width, base, block, out);
}

/// Inverse of [`pack_sorted`]: [`unpack`] a block of gaps and turn them into
/// values, `out[i]` being `base` plus the first `i + 1` gaps, each counting
/// `gap + 1`.
pub fn unpack_sorted(width: u32, base: Option<u32>, packed: &[u8], out: &mut [u32; BLOCK]) {
    check(width, packed.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    with_width!(width, simd::unpack_sorted(base, packed, out));
    #[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
    scalar::unpack_sorted(width, base, packed, out);
}

fn check(width: u32, packed: usize) {
    assert!(width <= 32, "bit width {width} out of range");
    assert_eq!(
        packed,
        packed_len(width),
        "packed block size for width {width}"
    );
}

#[cfg(test)]
mod tests {
    use rand::prelude::*;

    use super::*;

    /// Random blocks that fit `width`, plus the all-zeros and all-ones blocks.
    fn blocks(rng: &mut StdRng, width: u32) -> Vec<[u32; BLOCK]> {
        let mut blocks = vec![[0u32; BLOCK], [mask(width); BLOCK]];
        for _ in 0..8 {
            blocks.push(std::array::from_fn(|_| rng.random::<u32>() & mask(width)));
        }
        blocks
    }

    #[test]
    fn width_is_bits_to_hold_max() {
        let mut block = [0u32; BLOCK];
        assert_eq!(width(&block), 0);
        block[77] = 1;
        assert_eq!(width(&block), 1);
        block[3] = 0x1000;
        assert_eq!(width(&block), 13);
        block[0] = u32::MAX;
        assert_eq!(width(&block), 32);
    }

    #[test]
    fn scalar_roundtrips_every_width() {
        let mut rng = StdRng::seed_from_u64(1);
        for width in 0..=32 {
            for block in blocks(&mut rng, width) {
                let mut packed = vec![0u8; packed_len(width)];
                scalar::pack(width, &block, &mut packed);
                let mut back = [0u32; BLOCK];
                scalar::unpack(width, &packed, &mut back);
                assert_eq!(back, block, "width {width}");
            }
        }
    }

    /// The layout is the `bitpacking` crate's: lane `i % 4`, LSB first.
    #[test]
    fn layout_is_interleaved_lanes() {
        let block: [u32; BLOCK] = std::array::from_fn(|i| (i % 3) as u32);
        let mut packed = vec![0u8; packed_len(2)];
        scalar::pack(2, &block, &mut packed);
        // Lane 0 holds elements 0, 4, 8, … = 0, 1, 2, 0, 1, 2, …; its first
        // word is those sixteen 2-bit fields, LSB first.
        let lane0: u32 = (0..16)
            .map(|e| ((4 * e) % 3) as u32)
            .enumerate()
            .map(|(e, v)| v << (2 * e))
            .sum();
        assert_eq!(
            u32::from_le_bytes([packed[0], packed[1], packed[2], packed[3]]),
            lane0
        );
        let lane1: u32 = (0..16)
            .map(|e| ((4 * e + 1) % 3) as u32)
            .enumerate()
            .map(|(e, v)| v << (2 * e))
            .sum();
        assert_eq!(
            u32::from_le_bytes([packed[4], packed[5], packed[6], packed[7]]),
            lane1
        );
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    #[test]
    fn simd_matches_scalar_every_width() {
        let mut rng = StdRng::seed_from_u64(2);
        for width in 0..=32 {
            for block in blocks(&mut rng, width) {
                let mut expected = vec![0u8; packed_len(width)];
                scalar::pack(width, &block, &mut expected);
                let mut packed = vec![0u8; packed_len(width)];
                pack(width, &block, &mut packed);
                assert_eq!(packed, expected, "pack bytes, width {width}");
                let mut back = [0u32; BLOCK];
                unpack(width, &packed, &mut back);
                assert_eq!(back, block, "unpack, width {width}");
            }
            // Gaps narrower than 25 bits, so 128 of them plus a base fit in
            // a `u32` and the reference `delta::decode` cannot overflow.
            for gaps in blocks(&mut rng, width.min(24)) {
                for base in [None, Some(0), Some(rng.random::<u32>() >> 8)] {
                    let mut block = gaps;
                    crate::delta::decode(base, &mut block);
                    assert_eq!(
                        width_sorted(base, &block),
                        scalar::width_sorted(base, &block),
                        "width_sorted, width {width}, base {base:?}"
                    );
                    let mut expected = vec![0u8; packed_len(width)];
                    scalar::pack_sorted(width, base, &block, &mut expected);
                    let mut packed = vec![0u8; packed_len(width)];
                    pack_sorted(width, base, &block, &mut packed);
                    assert_eq!(
                        packed, expected,
                        "pack_sorted, width {width}, base {base:?}"
                    );
                    let mut back = [0u32; BLOCK];
                    unpack_sorted(width, base, &packed, &mut back);
                    assert_eq!(back, block, "unpack_sorted, width {width}, base {base:?}");
                    scalar::unpack_sorted(width, base, &packed, &mut back);
                    assert_eq!(
                        back, block,
                        "scalar unpack_sorted, width {width}, base {base:?}"
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "packed block size")]
    fn wrong_packed_len_panics() {
        let mut out = [0u32; BLOCK];
        unpack(3, &[0; 16], &mut out);
    }
}
