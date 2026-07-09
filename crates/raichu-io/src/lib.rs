//! # raichu-io — model I/O and cod3s interop
//!
//! JSON/TOML (de)serialization of native models, plus the **converter**
//! from cod3s's `cls`-tagged dumps (`ObjCOD3S.model_dump`) to the native
//! formalism: grouped connection interfaces → interfaces/ports,
//! references → in-ports + aggregation expressions, stringly-typed
//! indicator targets → typed, build-time-validated references.
//!
//! cod3s interop concepts live **only here** — `raichu-model` stays
//! native. Cross-validation runs through this converter, which is
//! itself a tested artefact (schema round-trip fixtures guard against
//! cod3s-dump drift).
//!
//! Populated in milestone M0, Phase 3.
