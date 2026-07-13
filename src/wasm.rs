//! WebAssembly bindings, built only for the `wasm32` target and published to
//! npm as `c2pa-ml`.
//!
//! Byte payloads map to/from JavaScript `Uint8Array`; a missing manifest (or an
//! unrecognized format) is thrown as a JS error.

use wasm_bindgen::prelude::*;

use crate::manifest::ManifestSource;

fn js_err(e: crate::Error) -> JsError {
    JsError::new(&e.to_string())
}

/// Embed a C2PA Manifest Store into a model, auto-detecting the format.
#[wasm_bindgen(js_name = embedManifest)]
pub fn embed_manifest(model: &[u8], store: &[u8]) -> Result<Vec<u8>, JsError> {
    crate::embed_manifest(model, &ManifestSource::embedded(store.to_vec())).map_err(js_err)
}

/// Embed a remote manifest URI into a model, auto-detecting the format.
#[wasm_bindgen(js_name = embedManifestRemote)]
pub fn embed_manifest_remote(model: &[u8], uri: &str) -> Result<Vec<u8>, JsError> {
    crate::embed_manifest(model, &ManifestSource::remote(uri)).map_err(js_err)
}

/// Embed both a Manifest Store and a remote URI into a model.
#[wasm_bindgen(js_name = embedManifestBoth)]
pub fn embed_manifest_both(model: &[u8], uri: &str, store: &[u8]) -> Result<Vec<u8>, JsError> {
    crate::embed_manifest(model, &ManifestSource::both(uri, store.to_vec())).map_err(js_err)
}

/// Read the embedded C2PA Manifest Store from a model.
#[wasm_bindgen(js_name = readManifest)]
pub fn read_manifest(model: &[u8]) -> Result<Vec<u8>, JsError> {
    crate::read_manifest(model).map_err(js_err)
}

/// Read the remote manifest URI from a model, if present.
#[wasm_bindgen(js_name = readManifestUri)]
pub fn read_manifest_uri(model: &[u8]) -> Result<Option<String>, JsError> {
    crate::read_manifest_uri(model).map_err(js_err)
}

/// Remove any C2PA metadata from a model.
#[wasm_bindgen(js_name = removeManifest)]
pub fn remove_manifest(model: &[u8]) -> Result<Vec<u8>, JsError> {
    crate::remove_manifest(model).map_err(js_err)
}

/// Detect the container format, returning `"GGUF"`, `"SafeTensors"`, `"ONNX"`,
/// or `undefined`.
#[wasm_bindgen(js_name = detectFormat)]
pub fn detect_format(model: &[u8]) -> Option<String> {
    crate::Format::detect(model).map(|f| f.name().to_string())
}

/// Return the canonical `c2pa.types.model.*` asset type string for the model's
/// format, or `undefined` if the format is unrecognized.
#[wasm_bindgen(js_name = modelType)]
pub fn model_type(model: &[u8]) -> Option<String> {
    crate::Format::detect(model).map(|f| f.model_type().as_str().to_string())
}
