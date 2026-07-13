<p align="center">
  <h1 align="center">c2pa-ml</h1>
  <p align="center">C2PA manifest embedding for AI/ML model container formats: GGUF, SafeTensors, and ONNX</p>
</p>

<p align="center">
  <a href="https://crates.io/crates/c2pa-ml"><img src="https://img.shields.io/crates/v/c2pa-ml.svg" alt="crates.io"></a>
  <a href="https://docs.rs/c2pa-ml"><img src="https://docs.rs/c2pa-ml/badge.svg" alt="docs.rs"></a>
  <a href="#license"><img src="https://img.shields.io/crates/l/c2pa-ml.svg" alt="License"></a>
</p>

## Overview

Associates a C2PA Manifest Store with an AI/ML model by writing it into the model container's own metadata slot, so the model stays loadable by its usual runtime. Three formats are supported:

| Format | Metadata slot | Manifest encoding |
|---|---|---|
| **GGUF** (llama.cpp) | typed key/value metadata | `c2pa.manifest` as a `UINT8` array (raw bytes) |
| **SafeTensors** | JSON header `__metadata__` | `c2pa.manifest` as Base64 |
| **ONNX** | protobuf `metadata_props` | `c2pa.manifest` as Base64 |

A remote (or side-car) manifest can instead be referenced by URI under `c2pa.manifest.uri`, or both an embedded store and a URI can be written together.

The [C2PA Technical Specification](https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html) defines no dedicated embedding method for model containers; a manifest embedded in a model declares what the asset is with the [asset type assertion](https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html#_asset_type). This crate provides the canonical `c2pa.types.model.*` strings for that assertion.

Zero dependencies.

## Quick Start

```toml
[dependencies]
c2pa-ml = "0.1"
```

### Embed a manifest

```rust
use c2pa_ml::{embed_manifest, ManifestSource};

let model: &[u8] = /* .gguf / .safetensors / .onnx bytes */;
let store: Vec<u8> = /* C2PA Manifest Store bytes */;

// Embed a Manifest Store directly (format is auto-detected)...
let signed = embed_manifest(model, &ManifestSource::embedded(store)).unwrap();

// ...or reference a remote manifest by URI...
let signed = embed_manifest(model, &ManifestSource::remote("https://example.com/m.c2pa")).unwrap();

// ...or both.
let signed = embed_manifest(model, &ManifestSource::both("https://example.com/m.c2pa", vec![/* ... */])).unwrap();
```

### Read a manifest

```rust
use c2pa_ml::{read_manifest, read_manifest_uri};

let store = read_manifest(&signed).unwrap();          // embedded Manifest Store bytes
let uri = read_manifest_uri(&signed).unwrap();        // Option<String>: active manifest URI
```

### Verify presence

```rust
use c2pa_ml::verify;

let report = verify(&signed).unwrap();
assert!(report.is_compliant());
// report.format, report.has_embedded_manifest, report.has_remote_uri
```

### Declare the asset type

```rust
use c2pa_ml::Format;

let model_type = Format::detect(&signed).unwrap().model_type();
assert_eq!(model_type.as_str(), "c2pa.types.model.onnx"); // for an ONNX model
```

### Explicit format

ONNX has no magic number, so auto-detection matches it last as a best-effort protobuf shape check. When the format is known in advance, use `embed_manifest_as`, or call the per-format module (`gguf`, `safetensors`, `onnx`) directly.

```rust
use c2pa_ml::{embed_manifest_as, Format, ManifestSource};

let signed = embed_manifest_as(model, Format::Onnx, &ManifestSource::embedded(store)).unwrap();
```

## Design

- The Manifest Store and/or manifest URI are stored under the reserved keys `c2pa.manifest` / `c2pa.manifest.uri` in the format's native metadata slot
- **GGUF**: metadata is re-serialized and the tensor-data region is re-padded to `general.alignment`; tensor-info offsets are relative to that region, so tensor data is never rewritten
- **SafeTensors**: only the JSON header is rewritten; each tensor's `data_offsets` are relative to the data block and stay valid
- **ONNX**: only the top-level protobuf field stream is rewritten; every other field (`ir_version`, `graph`, `opset_import`, …) is copied through verbatim
- Embedding replaces any existing C2PA entries; `remove_manifest` restores the model to its unembedded bytes

## Scope

This crate implements embedding and extraction only. Manifest construction, signing, and content (hard/soft) binding are out of scope; use the [official C2PA SDK](https://crates.io/crates/c2pa) to build and sign manifests. The `c2pa.hash.data` assertion should exclude the metadata region carrying the Manifest Store.

## Related Crates

| Crate | Description |
|---|---|
| [c2pa-fonts](https://crates.io/crates/c2pa-fonts) | OpenType/TrueType (SFNT) font embedding |
| [c2pa-warc](https://crates.io/crates/c2pa-warc) | WARC web archive embedding (ISO 28500) |
| [c2pa-zip](https://crates.io/crates/c2pa-zip) | ZIP-based (OCF) document embedding: EPUB, DOCX, ODT |
| [c2pa-structured-text](https://crates.io/crates/c2pa-structured-text) | Structured text embedding via ASCII armour delimiters |
| [c2pa-rs](https://crates.io/crates/c2pa) | Official C2PA SDK |

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.

Built by [WritersLogic](https://writerslogic.com)
