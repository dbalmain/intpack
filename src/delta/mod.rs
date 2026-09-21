//! The delta convention shared by every sorted codec: the first value as-is
//! (or relative to a carried `base`), then `gap - 1`, which a strictly
//! increasing list guarantees is non-negative and saves one bit per gap on
//! dense lists.

/// Four-lane SSE2 form of [`value`]: in-vector prefix sum of `gap + 1` plus
/// the previous group's last lane. The bitpack and Stream VByte kernels
/// share it.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
pub(crate) mod simd;

/// Delta-code `list` (strictly increasing) into `out`. `base` is the value
/// the first element is relative to, `None` for the head of a list.
pub fn encode(base: Option<u32>, list: &[u32], out: &mut Vec<u32>) {
    let mut prev = base;
    for &v in list {
        out.push(gap(prev, v));
        prev = Some(v);
    }
}

/// Inverse of [`encode`], in place.
pub fn decode(base: Option<u32>, deltas: &mut [u32]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    // SAFETY: this branch is compiled only when SSE2 is enabled.
    unsafe {
        simd::decode(base, deltas);
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
    decode_scalar(base, deltas);
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
fn decode_scalar(base: Option<u32>, deltas: &mut [u32]) {
    let mut prev = base;
    for d in deltas {
        let v = value(prev, *d);
        *d = v;
        prev = Some(v);
    }
}

/// One coded gap: `v` for the head of a list, else `v - prev - 1`.
#[inline]
pub fn gap(prev: Option<u32>, v: u32) -> u32 {
    match prev {
        None => v,
        Some(p) => v - p - 1,
    }
}

/// Inverse of [`gap`].
#[inline]
pub fn value(prev: Option<u32>, gap: u32) -> u32 {
    match prev {
        None => gap,
        Some(p) => p + gap + 1,
    }
}

/// `prev` for a loop that carries it as a plain `u32`: `u32::MAX` stands for
/// `None`, since `MAX + gap + 1` wraps to `gap`, which is the head-of-list
/// rule of [`value`] without a branch. Pair with [`value_after`].
#[inline]
pub fn carried(prev: Option<u32>) -> u32 {
    prev.unwrap_or(u32::MAX)
}

/// [`value`] on a [`carried`] `prev`.
#[inline]
pub fn value_after(prev: u32, gap: u32) -> u32 {
    prev.wrapping_add(gap).wrapping_add(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let list: Vec<u32> = vec![0, 1, 5, 6, 1000, u32::MAX];
        let mut coded = Vec::new();
        encode(None, &list, &mut coded);
        assert_eq!(coded, [0, 0, 3, 0, 993, u32::MAX - 1001]);
        decode(None, &mut coded);
        assert_eq!(coded, list);
    }

    #[test]
    fn carried_base() {
        let mut coded = Vec::new();
        encode(Some(9), &[10, 12], &mut coded);
        assert_eq!(coded, [0, 1]);
        decode(Some(9), &mut coded);
        assert_eq!(coded, [10, 12]);
    }

    #[test]
    fn carried_form_matches_value() {
        for prev in [None, Some(0), Some(9), Some(u32::MAX - 1001)] {
            for gap in [0, 1, 1000] {
                assert_eq!(
                    value_after(carried(prev), gap),
                    value(prev, gap),
                    "{prev:?} + {gap}"
                );
            }
        }
    }
}
