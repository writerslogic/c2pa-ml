//! The manifest payload written into a model, and the reserved metadata keys
//! it is stored under.

/// The reserved metadata key carrying an embedded C2PA Manifest Store.
pub(crate) const STORE_KEY: &str = "c2pa.manifest";

/// The reserved metadata key carrying a remote (or side-car) manifest URI.
pub(crate) const URI_KEY: &str = "c2pa.manifest.uri";

/// What to associate with a model: an embedded Manifest Store, a remote manifest
/// URI, or both.
///
/// Each supported format stores these in the same reserved keys
/// (`c2pa.manifest` / `c2pa.manifest.uri`) within its native metadata slot.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ManifestSource {
    /// URI of the active manifest. The caller fetches the bytes; this crate
    /// performs no network I/O.
    pub active_manifest_uri: Option<String>,
    /// Embedded C2PA Manifest Store bytes (JUMBF).
    pub manifest_store: Option<Vec<u8>>,
}

impl ManifestSource {
    /// Embed a Manifest Store directly in the model.
    pub fn embedded(manifest_store: Vec<u8>) -> Self {
        Self {
            active_manifest_uri: None,
            manifest_store: Some(manifest_store),
        }
    }

    /// Reference a remote (or side-car) manifest by URI.
    pub fn remote(uri: impl Into<String>) -> Self {
        Self {
            active_manifest_uri: Some(uri.into()),
            manifest_store: None,
        }
    }

    /// Embed a Manifest Store and record the active manifest URI.
    pub fn both(uri: impl Into<String>, manifest_store: Vec<u8>) -> Self {
        Self {
            active_manifest_uri: Some(uri.into()),
            manifest_store: Some(manifest_store),
        }
    }

    /// True when neither a store nor a URI is present.
    pub(crate) fn is_empty(&self) -> bool {
        self.active_manifest_uri.is_none() && self.manifest_store.is_none()
    }
}
