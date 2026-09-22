//! Ledger of code whose value was measured against one compiler and must be
//! re-measured when the compiler changes.
//!
//! A few attributes and one empty `asm!` in the cursors exist only to steer
//! LLVM's inlining and register allocation, each because a benchmark showed
//! the default choice costing nanoseconds per element. None of them is
//! correct or incorrect; each is a bet on one LLVM's heuristics, and a newer
//! rustc can make any of them useless or harmful without a test failing.
//! [`toolchain_moved_recheck_the_ledger`] fails when the compiling rustc is
//! not [`RECHECKED_WITH`], so a toolchain bump cannot land without someone
//! reading this list.
//!
//! # How to recheck
//!
//! Build `intpack-bench` against this crate, `cp` the binary to a fresh file
//! (its timings depend on the inode; see the bench README), and run the rows
//! named below with the item in place and with it removed — same session,
//! fresh copy each. Keep the item if it still pays; delete it, and its entry
//! here, if it is flat. Then set [`RECHECKED_WITH`] to the new version.
//!
//! ```sh
//! cd ../intpack-bench && cargo build --release && cp target/release/intpack-bench /tmp/ipb
//! /tmp/ipb bench --data data --out /tmp/res --budget 1 \
//!     --codec ip-pfor128 --codec ip-pfor128-skip --filter words.docs
//! ```
//!
//! # The ledger (rustc 1.95.0, LLVM 21.1.8, `-C target-cpu=native` on Zen 5)
//!
//! | item | where | measured effect if removed |
//! |---|---|---|
//! | `pin(target)` — empty `asm!` keeping the scan needle in this iteration | [`pfor128::SortedCursor::next_geq`](crate::pfor128) | LLVM hoists the sixteen-lane broadcast above the block loop and spills the `zmm` across `enter_block`; `ip-pfor128` words.docs ∩ 1:1 6.8 → 8.5 ns/e |
//! | `#[inline(always)]` on `value_at` | [`pfor128skip::SortedCursor`](crate::pfor128skip) | it falls out of line once the bitset walk is in it, which costs more than the call it replaced; trigrams ∩ 1:10 19.2 → 21.3 |
//! | `#[inline(never)]` on `load`, with the block check left in `enter` | same | `enter` falls out of line and every `next()` pays a call; trigrams ∩ 1:10 19.4 → 20.6 |
//! | `#[inline(never)]` on `search` | same | the same-block fast path in `next_geq` carries the table search and the enum match; that path as a whole took words ∩ 1:1 6.8 → 5.4 ns/e |
//! | `#[inline(never)]` on `write_bitset` | [`pfor128`](crate::pfor128) | inlined into the block encoder it costs lists with no bitset at all 9% of encode, 243 → 210 Mi/s (uniform-d0.01) |
//! | `#[cold] #[inline(never)]` on `Bitset::tail_word` | same | `Bitset::word` grows at every inlined site; no row moved, kept for size |
//!
//! The SIMD arms of `scan_geq` are not on this list: they are selected by
//! `cfg(target_feature)`, and `scan_agrees_on_random_blocks` runs every arm
//! the host can execute regardless of how the crate was compiled.

/// The rustc the ledger above was last measured with.
const RECHECKED_WITH: &str = "1.95.0";

#[cfg(test)]
mod tests {
    use super::RECHECKED_WITH;

    #[test]
    fn toolchain_moved_recheck_the_ledger() {
        let full = env!("INTPACK_RUSTC");
        let version = full.split_whitespace().nth(1).unwrap_or(full);
        assert_eq!(
            version, RECHECKED_WITH,
            "the compiler is now `{full}` and the inlining and asm bets in \
             src/toolchain.rs were measured with {RECHECKED_WITH}; re-measure \
             each row of the ledger and update RECHECKED_WITH"
        );
    }
}
