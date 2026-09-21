//! Integer sequence codecs for an inverted index.
//!
//! Every codec here is written from its paper or reference implementation,
//! not wrapped from a crate, so the algorithm is readable in one file. The
//! rule that keeps that true as the codecs get fast: **the format and the
//! algorithm live in plain scalar Rust; the tricks that make it fast live
//! behind a kernel boundary the algorithm calls through.** Bitpacking is the
//! worked example — [`bitpack`] has one scalar reference that *is* the
//! specification and an optional SIMD kernel that must agree with it bit for
//! bit. A reader who wants the algorithm reads the module; a reader who wants
//! the speed reads the kernel.
//!
//! # Contract shared by every codec
//!
//! * A list's **length** `n` is stored out of band by the caller (an index
//!   keeps it in the term dictionary), so `decode` and cursors receive it
//!   back. Codecs never spend bytes on it. Likewise the dataset **universe**
//!   (exclusive upper bound) for the codecs that need it.
//! * **Sorted** lists are strictly increasing `u32`; delta coding is the
//!   codec's job, and the shared convention is in [`delta`]: first value
//!   as-is, then `gap - 1`.
//! * **Unsorted** lists are arbitrary `u32`. A codec that only handles sorted
//!   input simply has no unsorted entry points.
//! * `decode` produces exactly the input list.
//! * Skip cursors implement [`Cursor`] with Lucene `advance` semantics.
//! * Formats are queried **in place** from the encoded bytes; opening a list
//!   for queries allocates nothing beyond a small cursor state.
//!
//! # Module map
//!
//! | module | shape | sorted | unsorted | cursor | get |
//! |---|---|:-:|:-:|:-:|:-:|
//! | [`vbyte`] | LEB128 varint | ✓ | ✓ | scan | — |
//! | [`streamvbyte`] | 2-bit control stream + data | ✓ | ✓ | — | — |
//! | [`bitpack`] | 128-int block kernels (the primitive) | | | | |
//! | [`pfor128`] | patched bitpacked 128-blocks + vbyte tail | ✓ | ✓ | scan | — |
//! | [`pfor128skip`] | pfor128 + skip table | ✓ | ✓ | ✓ | ✓ |
//! | [`fastpfor`] | patched frame-of-reference over blocks | ✓ | ✓ | — | — |
//! | [`elias_fano`] | EF with sampled select | ✓ | — | ✓ | ✓ |
//! | [`pef`] | partitioned EF | ✓ | — | ✓ | ✓ |

pub mod bitpack;
pub mod bits;
pub mod cursor;
pub mod delta;
pub mod elias_fano;
pub mod fastpfor;
pub mod pef;
pub mod pfor128;
pub mod pfor128skip;
pub mod streamvbyte;
pub mod vbyte;

pub use cursor::Cursor;
