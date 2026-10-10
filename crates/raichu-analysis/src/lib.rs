//! # raichu-analysis: analyses of a recorded sequence corpus
//!
//! The post-processing that reads what a simulation campaign recorded,
//! as opposed to the simulation itself, which lives in `raichu-core`:
//!
//! - [`sequence`]: the minimal-sequence reduction of a corpus of recorded
//!   trajectory [`Sequence`](raichu_core::Sequence)s
//!   (`group_sequences → filter_cycles → minimal_sequences`, chained by
//!   [`analyse`]);
//! - [`raw_sequences`]: the raw corpus format (`raichu.sequences`), read
//!   and written line by line, so a campaign too large for memory can be
//!   reduced later;
//! - [`importance`](mod@importance): the importance measures of each failure
//!   mode, component and declared group with respect to a feared event,
//!   computed from one campaign.
//!
//! The records themselves ([`Sequence`](raichu_core::Sequence),
//! [`SeqEvent`](raichu_core::SeqEvent), [`Provenance`](raichu_core::Provenance))
//! stay in `raichu-core`: they are what the engine writes. This crate
//! depends on `raichu-core` only; the Monte-Carlo driver and the
//! sequence-tree explorer depend on it where they reduce.
//!
//! These modules lived in `raichu-core` up to 0.58.0; the path changes are
//! listed in the migration guide (`docs/pycatshoo/migration-guide.md`,
//! section "Rust paths").

pub mod importance;
pub mod raw_sequences;
pub mod sequence;

pub use importance::{
    basic_events, importance, target_events, BasicEvent, BasicEventSelection, ComponentImportance,
    Cut, EventImportance, GroupImportance, ImportanceAnalysis, ImportanceError, ImportanceGroup,
    ImportanceReducer, Measures,
};
pub use raw_sequences::{
    read_raw_corpus, read_raw_sequences, write_raw_corpus, write_raw_sequences, ObservedCondition,
    RawCorpus, RawCorpusReader, RawCorpusWriter, RawHeader, RawObservation, RawSequencesError,
    RAW_SEQUENCES_FORMAT, RAW_SEQUENCES_VERSION,
};
pub use sequence::{
    analyse, clean, filter_cycles, group_sequences, minimal_sequences, SequenceReducer,
};
