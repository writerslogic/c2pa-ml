# c2pa-ml

_C2PA manifest embedding for AI/ML model container formats: GGUF, SafeTensors, and ONNX._

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

| Format | Metadata slot | Manifest encoding | Specified? |
|---|---|---|---|
| **ONNX** | protobuf `metadata_props` | `c2pa:manifest` as Base64 | yes |
| **SafeTensors** | JSON header `__metadata__` | `c2pa:manifest` as Base64 | yes |
| **GGUF** (llama.cpp) | typed key/value metadata | `c2pa:manifest` as a `UINT8` array (raw bytes) | **no** |

ONNX and SafeTensors each have a normative clause in the [C2PA Technical Specification](https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html), including the `c2pa:manifest` key, the hard binding, and a per-format `multipleManifests` failure code. This crate implements them as written.

**GGUF has no specified embedding method.** It is supported here as a crate extension, following the same shape so a dispatcher can treat all three alike — but there is no specified exclusion range for it, so [`binding::manifest_exclusion`](https://docs.rs/c2pa-ml) declines to invent one and returns `UnknownFormat`. Embedding and reading work; only the hard binding is unavailable.

A remote (or side-car) manifest can instead be referenced by URI under `c2pa:manifest.uri`, or both an embedded store and a URI can be written together. The specification defines no remote-URI key for these formats, so that too is a crate extension, namespaced to match.

A manifest embedded in a model should also declare what the asset is with the [asset type assertion](https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html#_asset_type); this crate provides the canonical `c2pa.types.model.*` strings for that.

Zero dependencies on native targets; the WebAssembly/npm build uses only `wasm-bindgen`.

## Quick Start

```toml
[dependencies]
c2pa-ml = "0.2"
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

## Other languages

The same API is published for JavaScript and Python from this crate, so it also serves Node, pnpm, and browser/bundler users.

### npm / pnpm (WebAssembly)

```bash
npm install c2pa-ml   # or: pnpm add c2pa-ml
```

```js
import { embedManifest, readManifest, detectFormat } from "c2pa-ml";

const signed = embedManifest(model, store); // Uint8Array in, Uint8Array out
const manifest = readManifest(signed);
```

### PyPI

```bash
pip install c2pa-ml
```

```python
import c2pa_ml

signed = c2pa_ml.embed_manifest(model, store)  # bytes in, bytes out
manifest = c2pa_ml.read_manifest(signed)
fmt = c2pa_ml.detect_format(model)             # "GGUF" | "SafeTensors" | "ONNX" | None
```

## Design

- The Manifest Store and/or manifest URI are stored under the reserved keys `c2pa:manifest` / `c2pa:manifest.uri` in the format's native metadata slot
- **GGUF**: metadata is re-serialized and the tensor-data region is re-padded to `general.alignment`; tensor-info offsets are relative to that region, so tensor data is never rewritten
- **SafeTensors**: only the JSON header is rewritten; each tensor's `data_offsets` are relative to the data block and stay valid
- **ONNX**: only the top-level protobuf field stream is rewritten; every other field (`ir_version`, `graph`, `opset_import`, …) is copied through verbatim
- Embedding replaces any existing C2PA entries; `remove_manifest` restores the model to its unembedded bytes

## Scope

This crate implements embedding and extraction only. Manifest construction, signing, and content (hard/soft) binding are out of scope; use the [official C2PA SDK](https://crates.io/crates/c2pa) to build and sign manifests. The `c2pa.hash.data` assertion should exclude the metadata region carrying the Manifest Store.

## Related Crates

Part of a family of single-purpose crates, one per C2PA embedding method. Each
is standalone and independently versioned.

| Crate | Description |
|---|---|
| [c2pa-structured-text](https://crates.io/crates/c2pa-structured-text) | Structured text: ASCII-armoured manifest in a comment or front matter |
| [c2pa-unstructured-text](https://crates.io/crates/c2pa-unstructured-text) | Unstructured text: invisible Unicode variation-selector run |
| [c2pa-html](https://crates.io/crates/c2pa-html) | HTML: `script` and `link` elements in the document head |
| [c2pa-http](https://crates.io/crates/c2pa-http) | HTTP: the `c2pa-manifest` `Link` header, with a Tower middleware |
| [c2pa-text-binding](https://crates.io/crates/c2pa-text-binding) | Soft binding and content fingerprinting for text assets |
| [c2pa-vtt](https://crates.io/crates/c2pa-vtt) | WebVTT caption and subtitle embedding |
| [c2pa-zip](https://crates.io/crates/c2pa-zip) | ZIP-based documents: EPUB, DOCX, ODT, OXPS |
| [c2pa-warc](https://crates.io/crates/c2pa-warc) | WARC web archive embedding (ISO 28500) |
| [c2pa-fonts](https://crates.io/crates/c2pa-fonts) | OpenType/TrueType (SFNT) font embedding |
| [c2pa](https://crates.io/crates/c2pa) | Official C2PA SDK |

## Security

Found a vulnerability? Please report it privately — see [SECURITY.md](./SECURITY.md).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.

Built by [WritersLogic](https://writerslogic.com)
