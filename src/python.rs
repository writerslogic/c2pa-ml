//! Python bindings, built with [maturin]/[PyO3] behind the `python` feature and
//! published to PyPI as `c2pa-ml`.
//!
//! Byte payloads map to/from Python `bytes`; a missing manifest (or an
//! unrecognized format) raises `ValueError`.
//!
//! [maturin]: https://www.maturin.rs/
//! [PyO3]: https://pyo3.rs/

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::manifest::ManifestSource;

fn map_err(e: crate::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Embed a C2PA Manifest Store into a model, auto-detecting the format.
#[pyfunction]
fn embed_manifest<'py>(
    py: Python<'py>,
    model: &[u8],
    store: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let out =
        crate::embed_manifest(model, &ManifestSource::embedded(store.to_vec())).map_err(map_err)?;
    Ok(PyBytes::new(py, &out))
}

/// Embed a remote manifest URI into a model, auto-detecting the format.
#[pyfunction]
fn embed_manifest_remote<'py>(
    py: Python<'py>,
    model: &[u8],
    uri: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let out = crate::embed_manifest(model, &ManifestSource::remote(uri)).map_err(map_err)?;
    Ok(PyBytes::new(py, &out))
}

/// Embed both a Manifest Store and a remote URI into a model.
#[pyfunction]
fn embed_manifest_both<'py>(
    py: Python<'py>,
    model: &[u8],
    uri: &str,
    store: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let out = crate::embed_manifest(model, &ManifestSource::both(uri, store.to_vec()))
        .map_err(map_err)?;
    Ok(PyBytes::new(py, &out))
}

/// Read the embedded C2PA Manifest Store from a model.
#[pyfunction]
fn read_manifest<'py>(py: Python<'py>, model: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let out = crate::read_manifest(model).map_err(map_err)?;
    Ok(PyBytes::new(py, &out))
}

/// Read the remote manifest URI from a model, or `None` if absent.
#[pyfunction]
fn read_manifest_uri(model: &[u8]) -> PyResult<Option<String>> {
    crate::read_manifest_uri(model).map_err(map_err)
}

/// Remove any C2PA metadata from a model.
#[pyfunction]
fn remove_manifest<'py>(py: Python<'py>, model: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let out = crate::remove_manifest(model).map_err(map_err)?;
    Ok(PyBytes::new(py, &out))
}

/// Detect the container format: `"GGUF"`, `"SafeTensors"`, `"ONNX"`, or `None`.
#[pyfunction]
fn detect_format(model: &[u8]) -> Option<String> {
    crate::Format::detect(model).map(|f| f.name().to_string())
}

/// The canonical `c2pa.types.model.*` asset type string for the model's format,
/// or `None` if the format is unrecognized.
#[pyfunction]
fn model_type(model: &[u8]) -> Option<String> {
    crate::Format::detect(model).map(|f| f.model_type().as_str().to_string())
}

#[pymodule]
fn c2pa_ml(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(embed_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(embed_manifest_remote, m)?)?;
    m.add_function(wrap_pyfunction!(embed_manifest_both, m)?)?;
    m.add_function(wrap_pyfunction!(read_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(read_manifest_uri, m)?)?;
    m.add_function(wrap_pyfunction!(remove_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(detect_format, m)?)?;
    m.add_function(wrap_pyfunction!(model_type, m)?)?;
    Ok(())
}
