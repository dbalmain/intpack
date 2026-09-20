//! Forward iteration with skip over one encoded sorted list.

/// Lucene's `advance` primitive, the one operation leapfrog intersection is
/// built on.
pub trait Cursor {
    /// Position on the smallest value `>= target` at or after the current
    /// element and return it, or `None` once exhausted. The cursor stays
    /// **on** that element: if the current element already satisfies the
    /// target the call returns it again without moving. Targets must be
    /// non-decreasing across calls.
    fn next_geq(&mut self, target: u32) -> Option<u32>;

    /// Move to the element after the current one and return it. A fresh
    /// cursor is positioned before the first element, so the first call
    /// returns it.
    fn next(&mut self) -> Option<u32>;
}
