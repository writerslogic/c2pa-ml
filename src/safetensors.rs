//! SafeTensors embedding.
//!
//! A SafeTensors file is a little-endian `u64` header length, a JSON header of
//! that length, then the raw tensor data. The header may carry a reserved
//! `__metadata__` object of string-to-string entries. A C2PA Manifest Store is
//! embedded there under `c2pa.manifest` as Base64 (the values are JSON strings);
//! a remote manifest URI is stored under `c2pa.manifest.uri`.
//!
//! Each tensor's `data_offsets` are relative to the start of the data block, so
//! rewriting the header never disturbs the tensor data.

use crate::base64;
use crate::error::Error;
use crate::json::{self, Value};
use crate::manifest::{ManifestSource, STORE_KEY, URI_KEY};

const METADATA_KEY: &str = "__metadata__";

/// True when `data` looks like a SafeTensors file (a plausible header length
/// pointing at a JSON object).
pub fn is_safetensors(data: &[u8]) -> bool {
    match header_bounds(data) {
        Ok((start, end)) => data[start..end]
            .iter()
            .find(|b| !b.is_ascii_whitespace())
            .is_some_and(|b| *b == b'{'),
        Err(_) => false,
    }
}

/// Embed a C2PA manifest into a SafeTensors file, replacing any existing C2PA
/// metadata entries.
pub fn embed(data: &[u8], source: &ManifestSource) -> Result<Vec<u8>, Error> {
    if source.is_empty() {
        return Err(Error::EmptySource);
    }
    let (mut header, body) = split(data)?;
    let meta = metadata_mut(&mut header)?;
    json::object_remove(meta, STORE_KEY);
    json::object_remove(meta, URI_KEY);
    if let Some(store) = &source.manifest_store {
        json::object_set(meta, STORE_KEY, Value::String(base64::encode(store)));
    }
    if let Some(uri) = &source.active_manifest_uri {
        json::object_set(meta, URI_KEY, Value::String(uri.clone()));
    }
    Ok(assemble(&header, body))
}

/// Read the embedded C2PA Manifest Store from a SafeTensors file.
pub fn read_store(data: &[u8]) -> Result<Vec<u8>, Error> {
    let (header, _) = split(data)?;
    let b64 = header
        .get(METADATA_KEY)
        .and_then(|m| m.get(STORE_KEY))
        .and_then(Value::as_str)
        .ok_or(Error::NotFound)?;
    base64::decode(b64).map_err(|e| Error::MalformedReference(e.to_string()))
}

/// Read the remote manifest URI from a SafeTensors file, if present.
pub fn read_uri(data: &[u8]) -> Result<Option<String>, Error> {
    let (header, _) = split(data)?;
    Ok(header
        .get(METADATA_KEY)
        .and_then(|m| m.get(URI_KEY))
        .and_then(Value::as_str)
        .map(str::to_string))
}

/// Remove any C2PA metadata entries from a SafeTensors file.
pub fn remove(data: &[u8]) -> Result<Vec<u8>, Error> {
    let (mut header, body) = split(data)?;
    if let Value::Object(entries) = &mut header {
        let drop_meta = if let Some((_, Value::Object(meta))) =
            entries.iter_mut().find(|(k, _)| k == METADATA_KEY)
        {
            json::object_remove(meta, STORE_KEY);
            json::object_remove(meta, URI_KEY);
            meta.is_empty()
        } else {
            false
        };
        if drop_meta {
            json::object_remove(entries, METADATA_KEY);
        }
    }
    Ok(assemble(&header, body))
}

/// The byte range `[start, end)` of the JSON header within `data`.
fn header_bounds(data: &[u8]) -> Result<(usize, usize), Error> {
    if data.len() < 8 {
        return Err(Error::Malformed(
            "file shorter than SafeTensors header".into(),
        ));
    }
    let n = u64::from_le_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ]) as usize;
    let end = 8usize
        .checked_add(n)
        .filter(|&e| e <= data.len())
        .ok_or_else(|| Error::Malformed("header length exceeds file".into()))?;
    Ok((8, end))
}

fn split(data: &[u8]) -> Result<(Value, &[u8]), Error> {
    let (start, end) = header_bounds(data)?;
    let text = std::str::from_utf8(&data[start..end])
        .map_err(|_| Error::Malformed("header is not UTF-8".into()))?;
    let header = json::parse(text).map_err(Error::Malformed)?;
    if !matches!(header, Value::Object(_)) {
        return Err(Error::Malformed("header is not a JSON object".into()));
    }
    Ok((header, &data[end..]))
}

fn metadata_mut(header: &mut Value) -> Result<&mut Vec<(String, Value)>, Error> {
    let entries = header
        .as_object_mut()
        .ok_or_else(|| Error::Malformed("header is not a JSON object".into()))?;
    if !entries.iter().any(|(k, _)| k == METADATA_KEY) {
        entries.push((METADATA_KEY.to_string(), Value::Object(Vec::new())));
    }
    let meta = entries
        .iter_mut()
        .find(|(k, _)| k == METADATA_KEY)
        .map(|(_, v)| v)
        .expect("just inserted");
    meta.as_object_mut()
        .ok_or_else(|| Error::Malformed("__metadata__ is not a JSON object".into()))
}

fn assemble(header: &Value, body: &[u8]) -> Vec<u8> {
    let text = json::to_string(header);
    let mut out = Vec::with_capacity(8 + text.len() + body.len());
    out.extend_from_slice(&(text.len() as u64).to_le_bytes());
    out.extend_from_slice(text.as_bytes());
    out.extend_from_slice(body);
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Build a SafeTensors file with one 8-byte `F32` tensor and eight bytes of
    /// data. `meta` is an optional pre-existing `__metadata__` fragment.
    pub fn sample_safetensors(meta: Option<&str>) -> Vec<u8> {
        let header = match meta {
            Some(m) => format!(
                r#"{{"__metadata__":{m},"t":{{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}}}"#
            ),
            None => r#"{"t":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#.to_string(),
        };
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[10, 20, 30, 40, 50, 60, 70, 80]);
        out
    }

    #[test]
    fn detects_format() {
        assert!(is_safetensors(&sample_safetensors(None)));
        assert!(!is_safetensors(b"GGUF...."));
        assert!(!is_safetensors(&[0, 0, 0, 0, 0, 0, 0, 0]));
    }

    #[test]
    fn embed_read_round_trip_no_prior_metadata() {
        let store = vec![1u8, 2, 3, 250, 0];
        let out = embed(
            &sample_safetensors(None),
            &ManifestSource::embedded(store.clone()),
        )
        .unwrap();
        assert_eq!(read_store(&out).unwrap(), store);
        assert_eq!(&out[out.len() - 8..], &[10, 20, 30, 40, 50, 60, 70, 80]);
    }

    #[test]
    fn preserves_existing_metadata_and_tensor_entry() {
        let out = embed(
            &sample_safetensors(Some(r#"{"format":"pt"}"#)),
            &ManifestSource::both("urn:x", vec![9]),
        )
        .unwrap();
        let (header, _) = split(&out).unwrap();
        assert_eq!(
            header
                .get("__metadata__")
                .and_then(|m| m.get("format"))
                .and_then(Value::as_str),
            Some("pt")
        );
        assert!(header.get("t").is_some());
        assert_eq!(read_uri(&out).unwrap().as_deref(), Some("urn:x"));
    }

    #[test]
    fn embed_replaces_existing() {
        let first = embed(
            &sample_safetensors(None),
            &ManifestSource::embedded(vec![1]),
        )
        .unwrap();
        let second = embed(&first, &ManifestSource::embedded(vec![2, 2])).unwrap();
        assert_eq!(read_store(&second).unwrap(), vec![2, 2]);
    }

    #[test]
    fn remove_restores_original_when_only_c2pa_metadata() {
        let out = embed(
            &sample_safetensors(None),
            &ManifestSource::embedded(vec![1, 2]),
        )
        .unwrap();
        let cleaned = remove(&out).unwrap();
        assert_eq!(cleaned, sample_safetensors(None));
    }

    #[test]
    fn remove_keeps_other_metadata() {
        let out = embed(
            &sample_safetensors(Some(r#"{"format":"pt"}"#)),
            &ManifestSource::embedded(vec![1]),
        )
        .unwrap();
        let cleaned = remove(&out).unwrap();
        assert!(matches!(read_store(&cleaned), Err(Error::NotFound)));
        let (header, _) = split(&cleaned).unwrap();
        assert_eq!(
            header
                .get("__metadata__")
                .and_then(|m| m.get("format"))
                .and_then(Value::as_str),
            Some("pt")
        );
    }

    #[test]
    fn empty_source_rejected() {
        assert!(matches!(
            embed(&sample_safetensors(None), &ManifestSource::default()),
            Err(Error::EmptySource)
        ));
    }

    #[test]
    fn rejects_truncated_header() {
        assert!(matches!(read_store(&[1, 2, 3]), Err(Error::Malformed(_))));
    }
}
