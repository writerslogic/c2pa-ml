//! Format detection and the format-dispatching public API.

use crate::asset_type::ModelType;
use crate::error::Error;
use crate::manifest::ManifestSource;
use crate::{gguf, onnx, safetensors};

/// A supported ML model container format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// GGUF (llama.cpp).
    Gguf,
    /// SafeTensors.
    SafeTensors,
    /// ONNX (`ModelProto`).
    Onnx,
}

impl Format {
    /// Detect the container format from the leading bytes.
    ///
    /// GGUF and SafeTensors are recognized structurally; ONNX has no magic
    /// number and is matched last as a best-effort protobuf shape check (see
    /// [`onnx::is_onnx`]), so it can yield a false positive on unrelated
    /// protobuf. When the format is known in advance, prefer
    /// [`embed_manifest_as`] over auto-detection.
    pub fn detect(data: &[u8]) -> Option<Format> {
        if gguf::is_gguf(data) {
            Some(Format::Gguf)
        } else if safetensors::is_safetensors(data) {
            Some(Format::SafeTensors)
        } else if onnx::is_onnx(data) {
            Some(Format::Onnx)
        } else {
            None
        }
    }

    /// A short human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Format::Gguf => "GGUF",
            Format::SafeTensors => "SafeTensors",
            Format::Onnx => "ONNX",
        }
    }

    /// The most specific C2PA [`ModelType`] for this container. GGUF and
    /// SafeTensors are framework-agnostic containers, so they map to the generic
    /// `c2pa.types.model`; ONNX maps to `c2pa.types.model.onnx`.
    pub fn model_type(self) -> ModelType {
        match self {
            Format::Gguf | Format::SafeTensors => ModelType::Generic,
            Format::Onnx => ModelType::Onnx,
        }
    }
}

/// A report on a model's C2PA embedding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    /// The detected container format.
    pub format: Format,
    /// The model carries an embedded Manifest Store.
    pub has_embedded_manifest: bool,
    /// The model carries a remote manifest URI.
    pub has_remote_uri: bool,
}

impl Report {
    /// A model is compliant when it references at least one manifest.
    pub fn is_compliant(&self) -> bool {
        self.has_embedded_manifest || self.has_remote_uri
    }
}

/// Embed a C2PA manifest into a model, auto-detecting the container format.
///
/// Returns [`Error::UnknownFormat`] when the format cannot be detected; call
/// [`embed_manifest_as`] with an explicit [`Format`] in that case.
pub fn embed_manifest(data: &[u8], source: &ManifestSource) -> Result<Vec<u8>, Error> {
    embed_manifest_as(data, detect(data)?, source)
}

/// Embed a C2PA manifest into a model of a known format.
pub fn embed_manifest_as(
    data: &[u8],
    format: Format,
    source: &ManifestSource,
) -> Result<Vec<u8>, Error> {
    match format {
        Format::Gguf => gguf::embed(data, source),
        Format::SafeTensors => safetensors::embed(data, source),
        Format::Onnx => onnx::embed(data, source),
    }
}

/// Read the embedded C2PA Manifest Store from a model, auto-detecting the format.
pub fn read_manifest(data: &[u8]) -> Result<Vec<u8>, Error> {
    match detect(data)? {
        Format::Gguf => gguf::read_store(data),
        Format::SafeTensors => safetensors::read_store(data),
        Format::Onnx => onnx::read_store(data),
    }
}

/// Read the remote manifest URI from a model, if present, auto-detecting the
/// format.
pub fn read_manifest_uri(data: &[u8]) -> Result<Option<String>, Error> {
    match detect(data)? {
        Format::Gguf => gguf::read_uri(data),
        Format::SafeTensors => safetensors::read_uri(data),
        Format::Onnx => onnx::read_uri(data),
    }
}

/// Remove any C2PA metadata from a model, auto-detecting the format.
pub fn remove_manifest(data: &[u8]) -> Result<Vec<u8>, Error> {
    match detect(data)? {
        Format::Gguf => gguf::remove(data),
        Format::SafeTensors => safetensors::remove(data),
        Format::Onnx => onnx::remove(data),
    }
}

/// Report on a model's C2PA embedding, auto-detecting the format.
pub fn verify(data: &[u8]) -> Result<Report, Error> {
    let format = detect(data)?;
    let has_embedded_manifest = match read_manifest(data) {
        Ok(_) => true,
        Err(Error::NotFound) => false,
        Err(e) => return Err(e),
    };
    let has_remote_uri = read_manifest_uri(data)?.is_some();
    Ok(Report {
        format,
        has_embedded_manifest,
        has_remote_uri,
    })
}

fn detect(data: &[u8]) -> Result<Format, Error> {
    Format::detect(data).ok_or(Error::UnknownFormat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::tests::sample_gguf;
    use crate::onnx::tests::sample_onnx;
    use crate::safetensors::tests::sample_safetensors;

    #[test]
    fn detects_each_format() {
        assert_eq!(Format::detect(&sample_gguf()), Some(Format::Gguf));
        assert_eq!(
            Format::detect(&sample_safetensors(None)),
            Some(Format::SafeTensors)
        );
        assert_eq!(Format::detect(&sample_onnx()), Some(Format::Onnx));
        assert_eq!(Format::detect(b"random bytes here!!"), None);
    }

    #[test]
    fn model_types() {
        assert_eq!(Format::Onnx.model_type().as_str(), "c2pa.types.model.onnx");
        assert_eq!(Format::Gguf.model_type().as_str(), "c2pa.types.model");
    }

    #[test]
    fn dispatch_round_trip_all_formats() {
        for data in [sample_gguf(), sample_safetensors(None), sample_onnx()] {
            let out = embed_manifest(&data, &ManifestSource::embedded(vec![1, 2, 3])).unwrap();
            assert_eq!(read_manifest(&out).unwrap(), vec![1, 2, 3]);
            let report = verify(&out).unwrap();
            assert!(report.is_compliant());
            assert!(report.has_embedded_manifest);
            let cleaned = remove_manifest(&out).unwrap();
            assert!(matches!(read_manifest(&cleaned), Err(Error::NotFound)));
        }
    }

    #[test]
    fn unknown_format_errors() {
        assert!(matches!(
            embed_manifest(b"nope", &ManifestSource::embedded(vec![1])),
            Err(Error::UnknownFormat)
        ));
    }
}
