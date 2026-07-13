// Copyright 2026 WritersLogic. All rights reserved.
// Licensed under the Apache License, Version 2.0 or the MIT license,
// at your option.

//! C2PA manifest embedding for AI/ML model container formats.
//!
//! A C2PA Manifest Store is associated with a model by writing it into the
//! format's own metadata slot, so the model stays loadable by its usual
//! runtime:
//!
//! - **GGUF** (llama.cpp) — a `c2pa.manifest` key/value metadata entry
//!   ([`gguf`]).
//! - **SafeTensors** — a `c2pa.manifest` entry in the JSON header's
//!   `__metadata__` ([`safetensors`]).
//! - **ONNX** — a `c2pa.manifest` entry in the protobuf `metadata_props`
//!   ([`onnx`]).
//!
//! A remote (or side-car) manifest can instead be referenced by URI under
//! `c2pa.manifest.uri`, or both an embedded store and a URI can be written
//! together (see [`ManifestSource`]).
//!
//! The top-level functions auto-detect the format:
//!
//! ```
//! use c2pa_ml::{embed_manifest, read_manifest, ManifestSource};
//! # fn demo(model: &[u8], store: Vec<u8>) -> Result<(), c2pa_ml::Error> {
//! let signed = embed_manifest(model, &ManifestSource::embedded(store))?;
//! let manifest = read_manifest(&signed)?;
//! # let _ = manifest;
//! # Ok(())
//! # }
//! ```
//!
//! # Asset type
//!
//! The C2PA core specification defines no dedicated embedding method for model
//! containers; a manifest embedded in a model declares what the asset is with
//! the asset type assertion. [`Format::model_type`] and [`asset_type::ModelType`]
//! provide the canonical `c2pa.types.model.*` strings for that assertion.
//!
//! # Scope
//!
//! This crate implements embedding and extraction only. Manifest construction,
//! signing, and content (hard/soft) binding are out of scope; use the
//! [official C2PA SDK](https://crates.io/crates/c2pa) to build and sign
//! manifests. The `c2pa.hash.data` assertion should exclude the metadata region
//! carrying the Manifest Store.
//!
//! Zero dependencies on native targets; the WebAssembly/npm build uses only
//! `wasm-bindgen`.

mod base64;
mod json;

pub mod asset_type;
pub mod gguf;
pub mod onnx;
pub mod safetensors;

mod error;
mod format;
mod manifest;

#[cfg(target_arch = "wasm32")]
mod wasm;

pub use asset_type::ModelType;
pub use error::Error;
pub use format::{
    embed_manifest, embed_manifest_as, read_manifest, read_manifest_uri, remove_manifest, verify,
    Format, Report,
};
pub use manifest::ManifestSource;
