//! GGUF (the llama.cpp model container) embedding.
//!
//! GGUF stores a block of typed key/value metadata at the head of the file,
//! ahead of the tensor information and tensor data. A C2PA Manifest Store is
//! embedded as a metadata entry under the reserved key `c2pa.manifest`, typed
//! as an array of `UINT8` so the raw JUMBF bytes survive byte-exact; a remote
//! manifest URI is stored under `c2pa.manifest.uri` as a `STRING`.
//!
//! Tensor-info offsets are relative to the start of the aligned tensor-data
//! region, so inserting metadata and re-padding to `general.alignment` leaves
//! every tensor valid without rewriting the tensor data.

use crate::error::Error;
use crate::manifest::{ManifestSource, STORE_KEY, URI_KEY};

const MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;

// GGUF metadata value types.
const T_UINT8: u32 = 0;
const T_UINT32: u32 = 4;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;

/// True when `data` begins with the GGUF magic.
pub fn is_gguf(data: &[u8]) -> bool {
    data.len() >= 4 && &data[..4] == MAGIC
}

/// Embed a C2PA manifest into a GGUF model, replacing any existing C2PA
/// metadata entries.
pub fn embed(data: &[u8], source: &ManifestSource) -> Result<Vec<u8>, Error> {
    if source.is_empty() {
        return Err(Error::EmptySource);
    }
    let mut file = Gguf::parse(data)?;
    file.kvs
        .retain(|kv| kv.key != STORE_KEY && kv.key != URI_KEY);
    if let Some(store) = &source.manifest_store {
        file.kvs.push(Kv {
            key: STORE_KEY.to_string(),
            value: encode_u8_array(store),
        });
    }
    if let Some(uri) = &source.active_manifest_uri {
        file.kvs.push(Kv {
            key: URI_KEY.to_string(),
            value: encode_string(uri),
        });
    }
    Ok(file.serialize())
}

/// Read the embedded C2PA Manifest Store from a GGUF model.
pub fn read_store(data: &[u8]) -> Result<Vec<u8>, Error> {
    let file = Gguf::parse(data)?;
    let kv = file
        .kvs
        .iter()
        .find(|kv| kv.key == STORE_KEY)
        .ok_or(Error::NotFound)?;
    decode_u8_array(&kv.value)
}

/// Read the remote manifest URI from a GGUF model, if present.
pub fn read_uri(data: &[u8]) -> Result<Option<String>, Error> {
    let file = Gguf::parse(data)?;
    match file.kvs.iter().find(|kv| kv.key == URI_KEY) {
        Some(kv) => decode_string(&kv.value).map(Some),
        None => Ok(None),
    }
}

/// Remove any C2PA metadata entries from a GGUF model.
pub fn remove(data: &[u8]) -> Result<Vec<u8>, Error> {
    let mut file = Gguf::parse(data)?;
    file.kvs
        .retain(|kv| kv.key != STORE_KEY && kv.key != URI_KEY);
    Ok(file.serialize())
}

struct Kv {
    key: String,
    /// The value encoding: the `u32` value type followed by its payload.
    value: Vec<u8>,
}

struct Gguf<'a> {
    version: u32,
    tensor_count: u64,
    kvs: Vec<Kv>,
    tensor_infos: &'a [u8],
    alignment: u64,
    data: &'a [u8],
}

impl<'a> Gguf<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        let mut c = Cursor::new(bytes);
        if c.take(4)? != MAGIC {
            return Err(Error::Malformed("not a GGUF file".into()));
        }
        let version = c.u32()?;
        if version != 2 && version != 3 {
            return Err(Error::Malformed(format!(
                "unsupported GGUF version {version}"
            )));
        }
        let tensor_count = c.u64()?;
        let kv_count = c.u64()?;

        let mut kvs = Vec::with_capacity(kv_count as usize);
        let mut alignment = DEFAULT_ALIGNMENT;
        for _ in 0..kv_count {
            let key = c.gguf_string()?;
            let start = c.pos;
            let vtype = c.u32()?;
            c.skip_value(vtype)?;
            let value = bytes[start..c.pos].to_vec();
            if key == "general.alignment" {
                if let Some(a) = read_u32_value(&value) {
                    if a != 0 {
                        alignment = a as u64;
                    }
                }
            }
            kvs.push(Kv { key, value });
        }

        let tensor_infos_start = c.pos;
        for _ in 0..tensor_count {
            c.gguf_string()?; // name
            let n_dims = c.u32()?;
            for _ in 0..n_dims {
                c.u64()?; // dimension
            }
            c.u32()?; // ggml type
            c.u64()?; // offset
        }
        let tensor_infos = &bytes[tensor_infos_start..c.pos];

        let data_start = align_up(c.pos as u64, alignment) as usize;
        if data_start > bytes.len() {
            return Err(Error::Malformed("tensor data offset out of range".into()));
        }
        let data = &bytes[data_start..];

        Ok(Gguf {
            version,
            tensor_count,
            kvs,
            tensor_infos,
            alignment,
            data,
        })
    }

    fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&self.tensor_count.to_le_bytes());
        out.extend_from_slice(&(self.kvs.len() as u64).to_le_bytes());
        for kv in &self.kvs {
            out.extend_from_slice(&(kv.key.len() as u64).to_le_bytes());
            out.extend_from_slice(kv.key.as_bytes());
            out.extend_from_slice(&kv.value);
        }
        out.extend_from_slice(self.tensor_infos);
        let data_start = align_up(out.len() as u64, self.alignment) as usize;
        out.resize(data_start, 0);
        out.extend_from_slice(self.data);
        out
    }
}

fn align_up(offset: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return offset;
    }
    offset.div_ceil(alignment) * alignment
}

fn encode_u8_array(bytes: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(16 + bytes.len());
    v.extend_from_slice(&T_ARRAY.to_le_bytes());
    v.extend_from_slice(&T_UINT8.to_le_bytes());
    v.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    v.extend_from_slice(bytes);
    v
}

fn decode_u8_array(value: &[u8]) -> Result<Vec<u8>, Error> {
    let mut c = Cursor::new(value);
    if c.u32()? != T_ARRAY || c.u32()? != T_UINT8 {
        return Err(Error::Malformed(
            "c2pa.manifest is not a UINT8 array".into(),
        ));
    }
    let count = c.u64()? as usize;
    Ok(c.take(count)?.to_vec())
}

fn encode_string(s: &str) -> Vec<u8> {
    let mut v = Vec::with_capacity(12 + s.len());
    v.extend_from_slice(&T_STRING.to_le_bytes());
    v.extend_from_slice(&(s.len() as u64).to_le_bytes());
    v.extend_from_slice(s.as_bytes());
    v
}

fn decode_string(value: &[u8]) -> Result<String, Error> {
    let mut c = Cursor::new(value);
    if c.u32()? != T_STRING {
        return Err(Error::Malformed("c2pa.manifest.uri is not a string".into()));
    }
    let len = c.u64()? as usize;
    let bytes = c.take(len)?;
    String::from_utf8(bytes.to_vec()).map_err(|_| Error::Malformed("URI is not UTF-8".into()))
}

fn read_u32_value(value: &[u8]) -> Option<u32> {
    let mut c = Cursor::new(value);
    if c.u32().ok()? != T_UINT32 {
        return None;
    }
    c.u32().ok()
}

fn scalar_size(vtype: u32) -> Option<usize> {
    match vtype {
        0 | 1 | 7 => Some(1), // UINT8, INT8, BOOL
        2..=3 => Some(2),     // UINT16, INT16
        4..=6 => Some(4),     // UINT32, INT32, FLOAT32
        10..=12 => Some(8),   // UINT64, INT64, FLOAT64
        _ => None,
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Cursor { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.bytes.len())
            .ok_or_else(|| Error::Malformed("unexpected end of GGUF data".into()))?;
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, Error> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, Error> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn gguf_string(&mut self) -> Result<String, Error> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| Error::Malformed("metadata key is not UTF-8".into()))
    }

    fn skip_value(&mut self, vtype: u32) -> Result<(), Error> {
        match vtype {
            T_STRING => {
                let len = self.u64()? as usize;
                self.take(len)?;
            }
            T_ARRAY => {
                let elem = self.u32()?;
                let count = self.u64()?;
                for _ in 0..count {
                    self.skip_value(elem)?;
                }
            }
            t => {
                let size = scalar_size(t)
                    .ok_or_else(|| Error::Malformed(format!("bad value type {t}")))?;
                self.take(size)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Build a minimal but valid GGUF v3 file: one string metadata entry and one
    /// 1-D `F32` tensor with four bytes of data.
    pub fn sample_gguf() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&3u32.to_le_bytes()); // version
        out.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        out.extend_from_slice(&1u64.to_le_bytes()); // kv_count
                                                    // kv: "general.name" = STRING "demo"
        let key = b"general.name";
        out.extend_from_slice(&(key.len() as u64).to_le_bytes());
        out.extend_from_slice(key);
        out.extend_from_slice(&encode_string("demo"));
        // tensor info: name "t", 1 dim = 1, type 0 (F32), offset 0
        let name = b"t";
        out.extend_from_slice(&(name.len() as u64).to_le_bytes());
        out.extend_from_slice(name);
        out.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        out.extend_from_slice(&1u64.to_le_bytes()); // dim[0]
        out.extend_from_slice(&0u32.to_le_bytes()); // ggml type
        out.extend_from_slice(&0u64.to_le_bytes()); // offset
                                                    // pad to alignment 32, then 4 bytes of tensor data
        out.resize(align_up(out.len() as u64, DEFAULT_ALIGNMENT) as usize, 0);
        out.extend_from_slice(&[1, 2, 3, 4]);
        out
    }

    #[test]
    fn detects_magic() {
        assert!(is_gguf(&sample_gguf()));
        assert!(!is_gguf(b"NOPE"));
    }

    #[test]
    fn round_trips_without_changes() {
        let g = sample_gguf();
        let parsed = Gguf::parse(&g).unwrap();
        assert_eq!(parsed.serialize(), g);
    }

    #[test]
    fn embed_read_store_round_trip() {
        let store = vec![0u8, 1, 2, 255, 0, 42];
        let out = embed(&sample_gguf(), &ManifestSource::embedded(store.clone())).unwrap();
        assert_eq!(read_store(&out).unwrap(), store);
        // Existing metadata and tensor data survive.
        assert_eq!(&out[out.len() - 4..], &[1, 2, 3, 4]);
    }

    #[test]
    fn embed_uri_and_both() {
        let out = embed(&sample_gguf(), &ManifestSource::remote("https://x/m.c2pa")).unwrap();
        assert_eq!(read_uri(&out).unwrap().as_deref(), Some("https://x/m.c2pa"));
        assert!(matches!(read_store(&out), Err(Error::NotFound)));

        let out = embed(
            &sample_gguf(),
            &ManifestSource::both("urn:x", vec![9, 8, 7]),
        )
        .unwrap();
        assert_eq!(read_store(&out).unwrap(), vec![9, 8, 7]);
        assert_eq!(read_uri(&out).unwrap().as_deref(), Some("urn:x"));
    }

    #[test]
    fn embed_replaces_existing() {
        let first = embed(&sample_gguf(), &ManifestSource::embedded(vec![1])).unwrap();
        let second = embed(&first, &ManifestSource::embedded(vec![2, 2])).unwrap();
        assert_eq!(read_store(&second).unwrap(), vec![2, 2]);
        let parsed = Gguf::parse(&second).unwrap();
        assert_eq!(
            parsed.kvs.iter().filter(|kv| kv.key == STORE_KEY).count(),
            1
        );
    }

    #[test]
    fn remove_strips_entries() {
        let out = embed(&sample_gguf(), &ManifestSource::both("urn:x", vec![1, 2])).unwrap();
        let cleaned = remove(&out).unwrap();
        assert!(matches!(read_store(&cleaned), Err(Error::NotFound)));
        assert_eq!(read_uri(&cleaned).unwrap(), None);
        assert_eq!(cleaned, sample_gguf());
    }

    #[test]
    fn empty_source_rejected() {
        assert!(matches!(
            embed(&sample_gguf(), &ManifestSource::default()),
            Err(Error::EmptySource)
        ));
    }

    #[test]
    fn honors_custom_alignment() {
        // Rebuild a sample with general.alignment = 16 and confirm re-embedding
        // keeps the tensor data intact.
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes()); // no tensors
        out.extend_from_slice(&1u64.to_le_bytes()); // one kv
        let key = b"general.alignment";
        out.extend_from_slice(&(key.len() as u64).to_le_bytes());
        out.extend_from_slice(key);
        out.extend_from_slice(&T_UINT32.to_le_bytes());
        out.extend_from_slice(&16u32.to_le_bytes());
        out.resize(align_up(out.len() as u64, 16) as usize, 0); // pad to data offset
        let parsed = Gguf::parse(&out).unwrap();
        assert_eq!(parsed.alignment, 16);
        let embedded = embed(&out, &ManifestSource::embedded(vec![7])).unwrap();
        assert_eq!(read_store(&embedded).unwrap(), vec![7]);
    }
}
