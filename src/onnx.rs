//! ONNX embedding.
//!
//! An ONNX model is a protobuf-encoded `ModelProto`. Its `metadata_props` field
//! (field number 14, a repeated `StringStringEntryProto`) is the model's
//! key/value metadata slot. A C2PA Manifest Store is embedded as a
//! `metadata_props` entry under `c2pa.manifest` as Base64 (the values are
//! protobuf strings); a remote manifest URI is stored under `c2pa.manifest.uri`.
//!
//! Embedding rewrites only the top-level field stream — every other field
//! (`ir_version`, `graph`, `opset_import`, …) is copied through verbatim.

use crate::base64;
use crate::error::Error;
use crate::manifest::{ManifestSource, STORE_KEY, URI_KEY};

const F_METADATA_PROPS: u64 = 14;
const F_KEY: u64 = 1;
const F_VALUE: u64 = 2;
const WIRE_LEN: u64 = 2;

/// Best-effort detection of an ONNX `ModelProto`. Protobuf has no magic number,
/// so this succeeds when the byte stream parses cleanly as a sequence of
/// top-level protobuf fields that includes `ir_version` (field 1) or `graph`
/// (field 7).
pub fn is_onnx(data: &[u8]) -> bool {
    match scan_fields(data) {
        Ok(fields) => fields.iter().any(|f| f.number == 1 || f.number == 7),
        Err(_) => false,
    }
}

/// Embed a C2PA manifest into an ONNX model, replacing any existing C2PA
/// metadata entries.
pub fn embed(data: &[u8], source: &ManifestSource) -> Result<Vec<u8>, Error> {
    if source.is_empty() {
        return Err(Error::EmptySource);
    }
    let fields = scan_fields(data)?;
    let mut out = Vec::with_capacity(data.len());
    for f in &fields {
        if f.number == F_METADATA_PROPS && entry_is_c2pa(&data[f.payload.clone()])? {
            continue;
        }
        out.extend_from_slice(&data[f.range.clone()]);
    }
    if let Some(store) = &source.manifest_store {
        out.extend_from_slice(&encode_entry(STORE_KEY, base64::encode(store).as_bytes()));
    }
    if let Some(uri) = &source.active_manifest_uri {
        out.extend_from_slice(&encode_entry(URI_KEY, uri.as_bytes()));
    }
    Ok(out)
}

/// Read the embedded C2PA Manifest Store from an ONNX model.
pub fn read_store(data: &[u8]) -> Result<Vec<u8>, Error> {
    let b64 = read_entry(data, STORE_KEY)?.ok_or(Error::NotFound)?;
    let text = String::from_utf8(b64).map_err(|_| Error::Malformed("store is not UTF-8".into()))?;
    base64::decode(&text).map_err(|e| Error::MalformedReference(e.to_string()))
}

/// Read the remote manifest URI from an ONNX model, if present.
pub fn read_uri(data: &[u8]) -> Result<Option<String>, Error> {
    match read_entry(data, URI_KEY)? {
        Some(bytes) => String::from_utf8(bytes)
            .map(Some)
            .map_err(|_| Error::Malformed("URI is not UTF-8".into())),
        None => Ok(None),
    }
}

/// Remove any C2PA metadata entries from an ONNX model.
pub fn remove(data: &[u8]) -> Result<Vec<u8>, Error> {
    let fields = scan_fields(data)?;
    let mut out = Vec::with_capacity(data.len());
    for f in &fields {
        if f.number == F_METADATA_PROPS && entry_is_c2pa(&data[f.payload.clone()])? {
            continue;
        }
        out.extend_from_slice(&data[f.range.clone()]);
    }
    Ok(out)
}

struct Field {
    number: u64,
    /// The full field bytes (tag + payload) within the source.
    range: std::ops::Range<usize>,
    /// The payload bytes (for length-delimited fields, the inner message).
    payload: std::ops::Range<usize>,
}

fn scan_fields(data: &[u8]) -> Result<Vec<Field>, Error> {
    let mut fields = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let start = pos;
        let (tag, next) = read_varint(data, pos)?;
        pos = next;
        let number = tag >> 3;
        let wire = tag & 7;
        let payload_start = pos;
        let payload_end = match wire {
            0 => read_varint(data, pos)?.1,
            1 => pos
                .checked_add(8)
                .filter(|&e| e <= data.len())
                .ok_or(trunc())?,
            2 => {
                let (len, after) = read_varint(data, pos)?;
                after
                    .checked_add(len as usize)
                    .filter(|&e| e <= data.len())
                    .ok_or(trunc())?
            }
            5 => pos
                .checked_add(4)
                .filter(|&e| e <= data.len())
                .ok_or(trunc())?,
            other => {
                return Err(Error::Malformed(format!("unsupported wire type {other}")));
            }
        };
        let payload = if wire == WIRE_LEN {
            let (_, after) = read_varint(data, payload_start)?;
            after..payload_end
        } else {
            payload_start..payload_end
        };
        pos = payload_end;
        fields.push(Field {
            number,
            range: start..pos,
            payload,
        });
    }
    Ok(fields)
}

/// Parse a `StringStringEntryProto` submessage into `(key, value)`.
fn parse_entry(msg: &[u8]) -> Result<(Vec<u8>, Vec<u8>), Error> {
    let mut key = Vec::new();
    let mut value = Vec::new();
    for f in scan_fields(msg)? {
        if f.number == F_KEY {
            key = msg[f.payload.clone()].to_vec();
        } else if f.number == F_VALUE {
            value = msg[f.payload.clone()].to_vec();
        }
    }
    Ok((key, value))
}

fn entry_is_c2pa(msg: &[u8]) -> Result<bool, Error> {
    let (key, _) = parse_entry(msg)?;
    Ok(key == STORE_KEY.as_bytes() || key == URI_KEY.as_bytes())
}

fn read_entry(data: &[u8], key: &str) -> Result<Option<Vec<u8>>, Error> {
    for f in scan_fields(data)? {
        if f.number == F_METADATA_PROPS {
            let (k, v) = parse_entry(&data[f.payload.clone()])?;
            if k == key.as_bytes() {
                return Ok(Some(v));
            }
        }
    }
    Ok(None)
}

fn encode_entry(key: &str, value: &[u8]) -> Vec<u8> {
    let mut msg = Vec::new();
    encode_string_field(&mut msg, F_KEY, key.as_bytes());
    encode_string_field(&mut msg, F_VALUE, value);
    let mut out = Vec::with_capacity(msg.len() + 8);
    encode_string_field(&mut out, F_METADATA_PROPS, &msg);
    out
}

fn encode_string_field(out: &mut Vec<u8>, number: u64, bytes: &[u8]) {
    write_varint(out, (number << 3) | WIRE_LEN);
    write_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn read_varint(data: &[u8], mut pos: usize) -> Result<(u64, usize), Error> {
    let mut value = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *data.get(pos).ok_or(trunc())?;
        pos += 1;
        if shift >= 64 {
            return Err(Error::Malformed("varint overflow".into()));
        }
        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok((value, pos));
        }
        shift += 7;
    }
}

fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

fn trunc() -> Error {
    Error::Malformed("unexpected end of ONNX protobuf".into())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Build a minimal `ModelProto`: `ir_version = 9` (field 1) and
    /// `producer_name = "demo"` (field 2).
    pub fn sample_onnx() -> Vec<u8> {
        let mut out = Vec::new();
        write_varint(&mut out, (1 << 3) | 0); // ir_version, varint
        write_varint(&mut out, 9);
        encode_string_field(&mut out, 2, b"demo"); // producer_name
        out
    }

    #[test]
    fn detects_shape() {
        assert!(is_onnx(&sample_onnx()));
        assert!(!is_onnx(b"GGUF\x03\x00\x00\x00"));
    }

    #[test]
    fn embed_read_round_trip() {
        let store = vec![0u8, 1, 2, 255, 7];
        let out = embed(&sample_onnx(), &ManifestSource::embedded(store.clone())).unwrap();
        assert_eq!(read_store(&out).unwrap(), store);
        // Original fields are preserved ahead of the appended entry.
        assert!(out.starts_with(&sample_onnx()));
    }

    #[test]
    fn embed_uri_and_both() {
        let out = embed(&sample_onnx(), &ManifestSource::remote("https://x/m.c2pa")).unwrap();
        assert_eq!(read_uri(&out).unwrap().as_deref(), Some("https://x/m.c2pa"));
        assert!(matches!(read_store(&out), Err(Error::NotFound)));

        let out = embed(&sample_onnx(), &ManifestSource::both("urn:x", vec![3, 3])).unwrap();
        assert_eq!(read_store(&out).unwrap(), vec![3, 3]);
        assert_eq!(read_uri(&out).unwrap().as_deref(), Some("urn:x"));
    }

    #[test]
    fn embed_replaces_existing() {
        let first = embed(&sample_onnx(), &ManifestSource::embedded(vec![1])).unwrap();
        let second = embed(&first, &ManifestSource::embedded(vec![2, 2])).unwrap();
        assert_eq!(read_store(&second).unwrap(), vec![2, 2]);
        let count = scan_fields(&second)
            .unwrap()
            .iter()
            .filter(|f| f.number == F_METADATA_PROPS)
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn remove_restores_original() {
        let out = embed(&sample_onnx(), &ManifestSource::both("urn:x", vec![1, 2])).unwrap();
        let cleaned = remove(&out).unwrap();
        assert_eq!(cleaned, sample_onnx());
    }

    #[test]
    fn preserves_foreign_metadata_prop() {
        let mut model = sample_onnx();
        model.extend_from_slice(&encode_entry("author", b"alice"));
        let out = embed(&model, &ManifestSource::embedded(vec![9])).unwrap();
        assert_eq!(read_store(&out).unwrap(), vec![9]);
        assert_eq!(read_entry(&out, "author").unwrap().unwrap(), b"alice");
        let cleaned = remove(&out).unwrap();
        assert_eq!(cleaned, model);
    }

    #[test]
    fn empty_source_rejected() {
        assert!(matches!(
            embed(&sample_onnx(), &ManifestSource::default()),
            Err(Error::EmptySource)
        ));
    }
}
