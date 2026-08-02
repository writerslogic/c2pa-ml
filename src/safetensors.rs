//! SafeTensors embedding.
//!
//! A SafeTensors file is a little-endian `u64` header length, a JSON header of
//! that length, then the raw tensor data. The header may carry a reserved
//! `__metadata__` object of string-to-string entries. A C2PA Manifest Store is
//! embedded there under `c2pa:manifest` as Base64 (the values are JSON strings);
//! a remote manifest URI is stored under `c2pa:manifest.uri`.
//!
//! Only one `c2pa:manifest` key shall be present; more than one is rejected with
//! `manifest.safetensors.multipleManifests`. The hard binding excludes the
//! Base64 value bytes only — see [`crate::binding`].
//!
//! Each tensor's `data_offsets` are relative to the start of the data block, so
//! rewriting the header never disturbs the tensor data.

use crate::base64;
use crate::error::Error;
use crate::format::Format;
use crate::json::{self, Value};
use crate::manifest::{ManifestSource, STORE_KEY, URI_KEY};
use std::ops::Range;

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
    // Via the span so the "only one key" rule and the header-length consistency
    // rule are enforced on every read, not just when a binding is computed.
    let span = store_value_span(data)?;
    let b64 = std::str::from_utf8(&data[span])
        .map_err(|_| Error::Malformed("store is not UTF-8".into()))?;
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

/// The byte spans, relative to `header`, of every string value stored under
/// `key` directly inside the `__metadata__` object.
///
/// A span covers the value's *content*: the bytes between the quotes, which is
/// exactly what the specification excludes from the data hash ("the
/// Base64-encoded value bytes"). Returns every occurrence so a caller can reject
/// a header carrying more than one.
///
/// This scans the serialized header rather than the parsed tree because the
/// parsed tree has no offsets, and the exclusion range is defined in bytes of
/// the file as stored.
fn metadata_value_spans(header: &[u8], key: &str) -> Vec<Range<usize>> {
    // One entry per open container: true for an object, false for an array.
    // Keys only exist in objects, so this is what tells a key from a value.
    let mut stack: Vec<bool> = Vec::new();
    let mut meta_depth: Option<usize> = None;
    let mut expect_key = false;
    let mut pending: Option<Range<usize>> = None;
    let mut out = Vec::new();
    let mut i = 0;

    while i < header.len() {
        match header[i] {
            b'{' => {
                if pending
                    .as_ref()
                    .is_some_and(|k| &header[k.clone()] == METADATA_KEY.as_bytes())
                {
                    meta_depth = Some(stack.len() + 1);
                }
                stack.push(true);
                expect_key = true;
                pending = None;
                i += 1;
            }
            b'[' => {
                stack.push(false);
                expect_key = false;
                pending = None;
                i += 1;
            }
            b'}' | b']' => {
                if meta_depth == Some(stack.len()) {
                    meta_depth = None;
                }
                stack.pop();
                expect_key = false;
                pending = None;
                i += 1;
            }
            b',' => {
                expect_key = stack.last().copied().unwrap_or(false);
                pending = None;
                i += 1;
            }
            b':' => {
                expect_key = false;
                i += 1;
            }
            b'"' => {
                let Some((content, next)) = scan_string(header, i) else {
                    // Unterminated string: nothing further can be trusted.
                    return out;
                };
                if expect_key {
                    pending = Some(content);
                    expect_key = false;
                } else {
                    if meta_depth == Some(stack.len())
                        && pending
                            .as_ref()
                            .is_some_and(|k| &header[k.clone()] == key.as_bytes())
                    {
                        out.push(content);
                    }
                    pending = None;
                }
                i = next;
            }
            _ => i += 1,
        }
    }
    out
}

/// Given the index of an opening quote, the span of the string's content and the
/// index just past the closing quote. `None` if the string is unterminated.
fn scan_string(bytes: &[u8], open: usize) -> Option<(Range<usize>, usize)> {
    let start = open + 1;
    let mut i = start;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 2,
            b'"' => return Some((start..i, i + 1)),
            _ => i += 1,
        }
    }
    None
}

/// The byte range of the Base64-encoded Manifest Store value within the file.
///
/// This is the single exclusion range the `c2pa.hash.data` assertion carries.
/// Offsets are absolute in `data`, so the header's own 8-byte length prefix is
/// already accounted for.
pub(crate) fn store_value_span(data: &[u8]) -> Result<Range<usize>, Error> {
    let (start, end) = header_bounds(data)?;
    check_header_length(data)?;
    let spans = metadata_value_spans(&data[start..end], STORE_KEY);
    match spans.len() {
        0 => Err(Error::NotFound),
        1 => {
            let s = &spans[0];
            Ok(start + s.start..start + s.end)
        }
        _ => Err(Error::MultipleManifests(Format::SafeTensors)),
    }
}

/// Reject a file whose 8-byte header length field disagrees with the actual
/// JSON header, which the specification requires a validator to treat as a
/// malformed data hash.
///
/// `header_bounds` already rejects a length that runs past the end of the file;
/// this additionally rejects one that stops short of the JSON header's real
/// extent, which would let the excluded range be pointed somewhere else.
pub(crate) fn check_header_length(data: &[u8]) -> Result<(), Error> {
    let (start, end) = header_bounds(data)?;
    let text = std::str::from_utf8(&data[start..end])
        .map_err(|_| Error::MalformedExclusion("header is not UTF-8".into()))?;
    json::parse(text).map_err(|e| {
        Error::MalformedExclusion(format!("header length field does not span valid JSON: {e}"))
    })?;
    Ok(())
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

    #[test]
    fn the_store_key_is_colon_namespaced_on_the_wire() {
        // The specification names `c2pa:manifest`. A dot here would embed into a
        // key no other implementation reads.
        let out = embed(
            &sample_safetensors(None),
            &ManifestSource::embedded(vec![1]),
        )
        .unwrap();
        assert!(out.windows(13).any(|w| w == b"c2pa:manifest"));
        assert!(!out.windows(13).any(|w| w == b"c2pa.manifest"));
    }

    #[test]
    fn the_value_span_points_at_the_base64_bytes() {
        let store = vec![9u8, 8, 7, 6, 5];
        let out = embed(
            &sample_safetensors(None),
            &ManifestSource::embedded(store.clone()),
        )
        .unwrap();
        let span = store_value_span(&out).unwrap();
        assert_eq!(&out[span], base64::encode(&store).as_bytes());
    }

    #[test]
    fn more_than_one_manifest_key_is_rejected() {
        // JSON permits duplicate keys in the raw text, so a foreign writer can
        // produce this even though this crate never does.
        let header = r#"{"__metadata__":{"c2pa:manifest":"aGk=","c2pa:manifest":"aGk="},"t":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.extend_from_slice(&[0u8; 8]);

        for result in [read_store(&file).err(), store_value_span(&file).err()] {
            let err = result.expect("a duplicate key must be rejected");
            assert!(matches!(err, Error::MultipleManifests(Format::SafeTensors)));
            assert_eq!(err.code(), Some("manifest.safetensors.multipleManifests"));
        }
    }

    #[test]
    fn a_header_length_field_that_disagrees_is_rejected() {
        // The specification requires `assertion.dataHash.malformed` when the
        // 8-byte length field does not match the actual JSON header, since it
        // would otherwise let the excluded range be pointed elsewhere.
        let good = embed(
            &sample_safetensors(None),
            &ManifestSource::embedded(vec![1]),
        )
        .unwrap();
        let declared = u64::from_le_bytes(good[..8].try_into().unwrap());

        let mut short = good.clone();
        short[..8].copy_from_slice(&(declared - 1).to_le_bytes());
        let err = store_value_span(&short).unwrap_err();
        assert!(matches!(err, Error::MalformedExclusion(_)));
        assert_eq!(err.code(), Some("assertion.dataHash.malformed"));

        // The untampered file is accepted, so the check is not vacuous.
        assert!(store_value_span(&good).is_ok());
    }

    #[test]
    fn a_matching_key_outside_the_metadata_object_is_not_the_manifest() {
        // A tensor named `c2pa:manifest` sits at header depth 1, not inside
        // `__metadata__`, and must not be mistaken for the store.
        let header = r#"{"c2pa:manifest":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.extend_from_slice(&[0u8; 8]);
        assert!(matches!(store_value_span(&file), Err(Error::NotFound)));
    }

    #[test]
    fn a_string_inside_an_array_is_not_mistaken_for_a_key() {
        // Arrays have no keys; the scanner must not treat an element as one.
        let header = r#"{"__metadata__":{"tags":["c2pa:manifest","x"],"c2pa:manifest":"aGk="}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let span = store_value_span(&file).unwrap();
        assert_eq!(&file[span], b"aGk=");
    }
}
