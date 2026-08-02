//! The `c2pa.hash.data` hard binding for ML model containers.
//!
//! # Coverage
//!
//! The hash is computed over the raw bytes of the complete file, with a single
//! exclusion range covering the **Base64-encoded value bytes** of the
//! `c2pa:manifest` entry — the bytes between the quotes for SafeTensors, the
//! `value` field payload for ONNX. Everything else, including the key itself and
//! the surrounding framing, is inside the hash.
//!
//! That is narrower than the exclusions some other formats use, and it has a
//! consequence: the encoded value's byte length must not change between hashing
//! and the final write, or every offset after it shifts. Hence the two-pass
//! flow.
//!
//! # Two-pass writing
//!
//! Unlike an HTML inline manifest — where the exclusion covers the whole element
//! and the hash can be computed before the manifest exists — here the framing
//! around the value is inside the hash, so the file must already have an entry
//! of the final size before it can be hashed:
//!
//! 1. [`reserve`] an entry sized to hold the Base64 of the eventual store.
//! 2. [`compute_data_hash`] over the reserved file.
//! 3. Sign the manifest using that hash.
//! 4. [`fill`] the signed store into the reserved entry, same encoded length.
//!
//! [`fill`] rejects a store whose Base64 differs in length from the placeholder,
//! because silently resizing would invalidate the hash that was just signed.
//!
//! # Formats
//!
//! ONNX and SafeTensors are specified. GGUF is a crate extension: it has no
//! specified embedding, so it has no specified exclusion either, and
//! [`manifest_exclusion`] declines to invent one.

use crate::base64;
use crate::error::Error;
use crate::format::{self, Format as Container};
use crate::manifest::ManifestSource;
use crate::{onnx, safetensors};

/// The assertion label for the hard binding.
pub const DATA_HASH_LABEL: &str = "c2pa.hash.data";

/// A byte range excluded from the data hash, matching the `EXCLUSION_RANGE-map`
/// CDDL (`start`, `length`). Offsets are into the file as stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exclusion {
    pub start: usize,
    pub length: usize,
}

impl Exclusion {
    fn end(&self) -> Option<usize> {
        self.start.checked_add(self.length)
    }
}

/// A C2PA-allowed hash algorithm for the data hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    Sha256,
    Sha384,
    Sha512,
}

impl Algorithm {
    /// The C2PA algorithm identifier used in the `alg` field.
    pub fn id(self) -> &'static str {
        match self {
            Algorithm::Sha256 => "sha256",
            Algorithm::Sha384 => "sha384",
            Algorithm::Sha512 => "sha512",
        }
    }

    pub fn from_id(id: &str) -> Result<Self, Error> {
        match id {
            "sha256" => Ok(Algorithm::Sha256),
            "sha384" => Ok(Algorithm::Sha384),
            "sha512" => Ok(Algorithm::Sha512),
            other => Err(Error::UnsupportedAlgorithm(other.to_string())),
        }
    }
}

/// A digest implementation. [`Sha2`] is the built-in one; the trait exists so a
/// caller can substitute an accelerated or host-provided digest without the
/// binding algorithm depending on either.
///
/// Model files run to gigabytes, so substituting here is not a theoretical
/// concern: the built-in implementation is portable, not vectorized.
pub trait Hasher {
    fn digest(&self, alg: Algorithm, data: &[u8]) -> Vec<u8>;
}

/// The built-in [`Hasher`]: SHA-256, SHA-384, and SHA-512 per FIPS 180-4,
/// implemented in-crate so the binding pulls in no dependency.
#[derive(Debug, Default, Clone, Copy)]
pub struct Sha2;

impl Hasher for Sha2 {
    fn digest(&self, alg: Algorithm, data: &[u8]) -> Vec<u8> {
        match alg {
            Algorithm::Sha256 => crate::sha2::sha256(data),
            Algorithm::Sha384 => crate::sha2::sha384(data),
            Algorithm::Sha512 => crate::sha2::sha512(data),
        }
    }
}

/// A computed `c2pa.hash.data` assertion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataHash {
    pub exclusions: Vec<Exclusion>,
    pub alg: String,
    pub hash: Vec<u8>,
    pub name: Option<String>,
}

impl DataHash {
    /// The assertion label, `c2pa.hash.data`.
    pub fn label(&self) -> &'static str {
        DATA_HASH_LABEL
    }

    /// Serialise to the JSON shape consumed when building a manifest, with the
    /// hash as standard Base64. The field set matches the `data-hash-map` CDDL.
    pub fn to_json(&self) -> String {
        let ranges: Vec<String> = self
            .exclusions
            .iter()
            .map(|e| format!("{{\"start\":{},\"length\":{}}}", e.start, e.length))
            .collect();
        let mut json = format!(
            "{{\"exclusions\":[{}],\"alg\":\"{}\",\"hash\":\"{}\"",
            ranges.join(","),
            self.alg,
            base64::encode(&self.hash)
        );
        if let Some(name) = &self.name {
            json.push_str(&format!(",\"name\":\"{name}\""));
        }
        json.push('}');
        json
    }
}

/// The single exclusion range covering the Base64-encoded Manifest Store value.
///
/// Fails with [`Error::UnknownFormat`] for GGUF: the format has no specified
/// C2PA embedding, so there is no specified exclusion to report.
pub fn manifest_exclusion(data: &[u8]) -> Result<Exclusion, Error> {
    let span = match format::detect(data)? {
        Container::Onnx => onnx::store_value_span(data)?,
        Container::SafeTensors => safetensors::store_value_span(data)?,
        Container::Gguf => return Err(Error::UnknownFormat),
    };
    Ok(Exclusion {
        start: span.start,
        length: span.len(),
    })
}

/// Remove `exclusions` from `data`, validating that they are ordered,
/// non-overlapping, and within bounds.
pub fn apply_exclusions(data: &[u8], exclusions: &[Exclusion]) -> Result<Vec<u8>, Error> {
    let mut cursor = 0usize;
    let mut out = Vec::with_capacity(data.len());
    for ex in exclusions {
        let end = ex
            .end()
            .ok_or_else(|| Error::MalformedExclusion("range overflows".into()))?;
        if ex.start < cursor {
            return Err(Error::MalformedExclusion(
                "ranges are out of order or overlapping".into(),
            ));
        }
        if end > data.len() {
            return Err(Error::MalformedExclusion(
                "range extends past the file".into(),
            ));
        }
        out.extend_from_slice(&data[cursor..ex.start]);
        cursor = end;
    }
    out.extend_from_slice(&data[cursor..]);
    Ok(out)
}

/// Compute the hard binding for `data`: locate the Base64 manifest value,
/// exclude it, and hash the rest of the file.
pub fn compute_data_hash(
    data: &[u8],
    alg: Algorithm,
    hasher: &impl Hasher,
) -> Result<DataHash, Error> {
    let exclusion = manifest_exclusion(data)?;
    let covered = apply_exclusions(data, &[exclusion])?;
    Ok(DataHash {
        exclusions: vec![exclusion],
        alg: alg.id().to_string(),
        hash: hasher.digest(alg, &covered),
        name: None,
    })
}

/// Verify a `c2pa.hash.data` binding against `data`, following the validator
/// procedure: apply the assertion's own exclusion ranges, recompute, compare.
///
/// The assertion's range must match the located manifest value. One that
/// excludes some other span would otherwise hash a file the manifest does not
/// describe.
pub fn verify_data_hash(
    data: &[u8],
    data_hash: &DataHash,
    hasher: &impl Hasher,
) -> Result<(), Error> {
    let alg = Algorithm::from_id(&data_hash.alg)?;
    let located = manifest_exclusion(data)?;
    if !data_hash.exclusions.contains(&located) {
        return Err(Error::MalformedExclusion(
            "assertion does not exclude the located manifest value".into(),
        ));
    }
    let covered = apply_exclusions(data, &data_hash.exclusions)?;
    if hasher.digest(alg, &covered) == data_hash.hash {
        Ok(())
    } else {
        Err(Error::HashMismatch)
    }
}

/// Reserve a `c2pa:manifest` entry large enough to hold `store_size` bytes of
/// Manifest Store, so the file can be hashed before the manifest is signed.
///
/// The placeholder is `store_size` zero bytes, whose Base64 is the same length
/// as that of any other store of the same size — which is what keeps the
/// exclusion range stable across [`fill`].
pub fn reserve(data: &[u8], store_size: usize) -> Result<Vec<u8>, Error> {
    crate::embed_manifest(data, &ManifestSource::embedded(vec![0u8; store_size]))
}

/// Replace a reserved placeholder with the signed Manifest Store.
///
/// Fails with [`Error::MalformedExclusion`] if the store's Base64 length differs
/// from the placeholder's, since that would shift every byte after it and
/// invalidate the hash that was just signed over.
pub fn fill(data: &[u8], store: &[u8]) -> Result<Vec<u8>, Error> {
    let reserved = manifest_exclusion(data)?;
    let encoded = base64::encode(store);
    if encoded.len() != reserved.length {
        return Err(Error::MalformedExclusion(format!(
            "signed manifest encodes to {} bytes but {} were reserved",
            encoded.len(),
            reserved.length
        )));
    }
    let mut out = data.to_vec();
    out[reserved.start..reserved.start + reserved.length].copy_from_slice(encoded.as_bytes());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::tests::sample_onnx;
    use crate::safetensors::tests::sample_safetensors;

    const STORE: &[u8] = b"manifest-store-bytes";

    /// Both specified containers, so every test runs against each.
    fn specified() -> Vec<(&'static str, Vec<u8>)> {
        vec![
            ("onnx", sample_onnx()),
            ("safetensors", sample_safetensors(None)),
        ]
    }

    fn embedded(base: &[u8], store: &[u8]) -> Vec<u8> {
        crate::embed_manifest(base, &ManifestSource::embedded(store.to_vec())).unwrap()
    }

    #[test]
    fn the_exclusion_covers_exactly_the_base64_value() {
        for (name, base) in specified() {
            let data = embedded(&base, STORE);
            let ex = manifest_exclusion(&data).unwrap();
            assert_eq!(
                &data[ex.start..ex.start + ex.length],
                base64::encode(STORE).as_bytes(),
                "{name}"
            );
        }
    }

    #[test]
    fn the_key_and_framing_stay_inside_the_hash() {
        // Only the value is excluded, so the key bytes must survive exclusion.
        for (name, base) in specified() {
            let data = embedded(&base, STORE);
            let ex = manifest_exclusion(&data).unwrap();
            let covered = apply_exclusions(&data, &[ex]).unwrap();
            assert!(
                covered.windows(13).any(|w| w == b"c2pa:manifest"),
                "{name}: the key was excluded along with the value"
            );
        }
    }

    #[test]
    fn compute_then_verify_round_trips() {
        for (name, base) in specified() {
            let data = embedded(&base, STORE);
            let dh = compute_data_hash(&data, Algorithm::Sha256, &Sha2).unwrap();
            assert_eq!(dh.alg, "sha256");
            assert_eq!(dh.label(), "c2pa.hash.data");
            assert!(verify_data_hash(&data, &dh, &Sha2).is_ok(), "{name}");
        }
    }

    #[test]
    fn every_algorithm_round_trips() {
        for (name, base) in specified() {
            let data = embedded(&base, STORE);
            for alg in [Algorithm::Sha256, Algorithm::Sha384, Algorithm::Sha512] {
                let dh = compute_data_hash(&data, alg, &Sha2).unwrap();
                assert!(
                    verify_data_hash(&data, &dh, &Sha2).is_ok(),
                    "{name} {alg:?}"
                );
            }
        }
    }

    #[test]
    fn tampering_with_the_model_breaks_the_binding() {
        // Change a tensor data byte, which is outside the exclusion.
        let base = sample_safetensors(None);
        let data = embedded(&base, STORE);
        let dh = compute_data_hash(&data, Algorithm::Sha256, &Sha2).unwrap();

        let mut tampered = data.clone();
        let last = tampered.len() - 1;
        tampered[last] ^= 0xFF;
        assert!(matches!(
            verify_data_hash(&tampered, &dh, &Sha2),
            Err(Error::HashMismatch)
        ));
    }

    #[test]
    fn changing_the_manifest_alone_does_not_break_the_binding() {
        // The value is excluded, so swapping a same-length store keeps the hash.
        let base = sample_safetensors(None);
        let a = embedded(&base, b"aaaaaaaaaaaaaaaaaaaa");
        let b = embedded(&base, b"bbbbbbbbbbbbbbbbbbbb");
        let ha = compute_data_hash(&a, Algorithm::Sha256, &Sha2).unwrap();
        let hb = compute_data_hash(&b, Algorithm::Sha256, &Sha2).unwrap();
        assert_eq!(ha.hash, hb.hash);
        assert_eq!(ha.exclusions, hb.exclusions);
    }

    #[test]
    fn the_two_pass_flow_produces_a_binding_that_verifies() {
        for (name, base) in specified() {
            // 1. Reserve, 2. hash, 3. "sign", 4. fill.
            let reserved = reserve(&base, STORE.len()).unwrap();
            let dh = compute_data_hash(&reserved, Algorithm::Sha256, &Sha2).unwrap();
            let filled = fill(&reserved, STORE).unwrap();

            // The hash signed over the reservation still verifies after filling.
            assert!(verify_data_hash(&filled, &dh, &Sha2).is_ok(), "{name}");
            assert_eq!(crate::read_manifest(&filled).unwrap(), STORE, "{name}");
        }
    }

    #[test]
    fn fill_rejects_a_store_that_would_shift_the_offsets() {
        let reserved = reserve(&sample_onnx(), 20).unwrap();
        assert!(matches!(
            fill(&reserved, b"this store is a different length entirely"),
            Err(Error::MalformedExclusion(_))
        ));
        // A store of the reserved size is accepted.
        assert!(fill(&reserved, &[7u8; 20]).is_ok());
    }

    #[test]
    fn an_exclusion_that_is_not_the_manifest_value_is_rejected() {
        let data = embedded(&sample_onnx(), STORE);
        let mut dh = compute_data_hash(&data, Algorithm::Sha256, &Sha2).unwrap();
        dh.exclusions = vec![Exclusion {
            start: 0,
            length: 4,
        }];
        assert!(matches!(
            verify_data_hash(&data, &dh, &Sha2),
            Err(Error::MalformedExclusion(_))
        ));
    }

    #[test]
    fn malformed_ranges_are_rejected() {
        let data = embedded(&sample_onnx(), STORE);
        let out_of_order = [
            Exclusion {
                start: 10,
                length: 5,
            },
            Exclusion {
                start: 5,
                length: 5,
            },
        ];
        assert!(matches!(
            apply_exclusions(&data, &out_of_order),
            Err(Error::MalformedExclusion(_))
        ));
        assert!(matches!(
            apply_exclusions(
                &data,
                &[Exclusion {
                    start: 0,
                    length: data.len() + 1
                }]
            ),
            Err(Error::MalformedExclusion(_))
        ));
        assert!(matches!(
            apply_exclusions(
                &data,
                &[Exclusion {
                    start: usize::MAX,
                    length: 1
                }]
            ),
            Err(Error::MalformedExclusion(_))
        ));
    }

    #[test]
    fn unsupported_algorithm_is_reported() {
        let data = embedded(&sample_onnx(), STORE);
        let mut dh = compute_data_hash(&data, Algorithm::Sha256, &Sha2).unwrap();
        dh.alg = "sha1".into();
        assert!(matches!(
            verify_data_hash(&data, &dh, &Sha2),
            Err(Error::UnsupportedAlgorithm(_))
        ));
    }

    #[test]
    fn binding_a_model_without_a_manifest_reports_not_found() {
        for (name, base) in specified() {
            assert!(
                matches!(
                    compute_data_hash(&base, Algorithm::Sha256, &Sha2),
                    Err(Error::NotFound)
                ),
                "{name}"
            );
        }
    }

    #[test]
    fn gguf_has_no_specified_exclusion() {
        // GGUF embedding is a crate extension, so there is no specified
        // exclusion to report and none is invented.
        let data = embedded(&crate::gguf::tests::sample_gguf(), STORE);
        assert!(matches!(
            manifest_exclusion(&data),
            Err(Error::UnknownFormat)
        ));
        // Embedding and reading still work; only the binding is unavailable.
        assert_eq!(crate::read_manifest(&data).unwrap(), STORE);
    }

    #[test]
    fn algorithm_ids_round_trip() {
        for alg in [Algorithm::Sha256, Algorithm::Sha384, Algorithm::Sha512] {
            assert_eq!(Algorithm::from_id(alg.id()).unwrap(), alg);
        }
        assert!(matches!(
            Algorithm::from_id("md5"),
            Err(Error::UnsupportedAlgorithm(_))
        ));
    }

    #[test]
    fn json_shape_matches_the_data_hash_map() {
        let dh = DataHash {
            exclusions: vec![Exclusion {
                start: 73,
                length: 114,
            }],
            alg: "sha256".into(),
            hash: vec![0xDE, 0xAD, 0xBE, 0xEF],
            name: None,
        };
        assert_eq!(
            dh.to_json(),
            r#"{"exclusions":[{"start":73,"length":114}],"alg":"sha256","hash":"3q2+7w=="}"#
        );
    }
}
