//! End-to-end use of the public API, as a consumer sees it.
//!
//! The unit tests reach into private module fixtures; these build their assets
//! from the format specifications directly and use only the exported API, so an
//! accidental change to the public surface shows up here.

use c2pa_ml::binding::{
    compute_data_hash, fill, manifest_exclusion, reserve, verify_data_hash, Algorithm, Sha2,
};
use c2pa_ml::{embed_manifest, read_manifest, read_manifest_uri, remove_manifest, Error, Format};

const STORE: &[u8] = b"\x00\x01\x02manifest-store-bytes\xFF";

/// A SafeTensors file: an 8-byte little-endian header length, that many bytes
/// of JSON, then the tensor data.
fn safetensors() -> Vec<u8> {
    let header = r#"{"t":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&[10, 20, 30, 40, 50, 60, 70, 80]);
    out
}

#[test]
fn safetensors_embed_read_and_remove() {
    let model = safetensors();
    let embedded =
        embed_manifest(&model, &c2pa_ml::ManifestSource::embedded(STORE.to_vec())).expect("embed");
    assert_eq!(read_manifest(&embedded).unwrap(), STORE);

    // Tensor data is untouched: `data_offsets` are relative to the data block.
    assert_eq!(
        &embedded[embedded.len() - 8..],
        &[10, 20, 30, 40, 50, 60, 70, 80]
    );

    assert_eq!(remove_manifest(&embedded).unwrap(), model);
}

#[test]
fn the_wire_key_is_colon_namespaced() {
    // The specification names `c2pa:manifest`. A dot would be a key no other
    // implementation reads.
    let embedded = embed_manifest(
        &safetensors(),
        &c2pa_ml::ManifestSource::embedded(STORE.to_vec()),
    )
    .unwrap();
    assert!(embedded.windows(13).any(|w| w == b"c2pa:manifest"));
    assert!(!embedded.windows(13).any(|w| w == b"c2pa.manifest"));
}

#[test]
fn the_exclusion_covers_exactly_the_base64_value() {
    let embedded = embed_manifest(
        &safetensors(),
        &c2pa_ml::ManifestSource::embedded(STORE.to_vec()),
    )
    .unwrap();
    let ex = manifest_exclusion(&embedded).unwrap();
    let covered = &embedded[ex.start..ex.start + ex.length];
    // Base64 alphabet only — the quotes and the key are outside the exclusion.
    assert!(covered
        .iter()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'=')));
}

#[test]
fn the_two_pass_flow_produces_a_binding_that_survives_filling() {
    // Reserve, hash, "sign", fill: the exclusion covers only the value, so the
    // surrounding framing is inside the hash and must not shift.
    let reserved = reserve(&safetensors(), STORE.len()).unwrap();
    let binding = compute_data_hash(&reserved, Algorithm::Sha256, &Sha2).unwrap();
    let filled = fill(&reserved, STORE).unwrap();

    assert!(verify_data_hash(&filled, &binding, &Sha2).is_ok());
    assert_eq!(read_manifest(&filled).unwrap(), STORE);
}

#[test]
fn filling_a_differently_sized_store_is_refused() {
    let reserved = reserve(&safetensors(), STORE.len()).unwrap();
    assert!(matches!(
        fill(
            &reserved,
            b"a different length entirely, much longer than reserved"
        ),
        Err(Error::MalformedExclusion(_))
    ));
}

#[test]
fn tampering_with_tensor_data_breaks_the_binding() {
    let embedded = embed_manifest(
        &safetensors(),
        &c2pa_ml::ManifestSource::embedded(STORE.to_vec()),
    )
    .unwrap();
    let binding = compute_data_hash(&embedded, Algorithm::Sha256, &Sha2).unwrap();

    let mut tampered = embedded.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 0xFF;
    assert!(matches!(
        verify_data_hash(&tampered, &binding, &Sha2),
        Err(Error::HashMismatch)
    ));
}

#[test]
fn a_remote_uri_round_trips() {
    let uri = "https://fabrikam.example/m.c2pa";
    let embedded = embed_manifest(&safetensors(), &c2pa_ml::ManifestSource::remote(uri)).unwrap();
    assert_eq!(read_manifest_uri(&embedded).unwrap().as_deref(), Some(uri));
}

#[test]
fn a_header_length_that_disagrees_is_rejected() {
    // The specification requires `assertion.dataHash.malformed` when the 8-byte
    // length field does not match the actual JSON header.
    let good = embed_manifest(
        &safetensors(),
        &c2pa_ml::ManifestSource::embedded(STORE.to_vec()),
    )
    .unwrap();
    let declared = u64::from_le_bytes(good[..8].try_into().unwrap());
    let mut short = good.clone();
    short[..8].copy_from_slice(&(declared - 1).to_le_bytes());

    let err = manifest_exclusion(&short).unwrap_err();
    assert_eq!(err.code(), Some("assertion.dataHash.malformed"));
}

#[test]
fn format_detection_matches_the_container() {
    assert_eq!(Format::detect(&safetensors()), Some(Format::SafeTensors));
    assert_eq!(Format::detect(b"GGUF\x03\x00\x00\x00"), Some(Format::Gguf));
    assert_eq!(Format::detect(b"\x00\x00\x00"), None);
}

#[test]
fn every_allowed_algorithm_round_trips() {
    let embedded = embed_manifest(
        &safetensors(),
        &c2pa_ml::ManifestSource::embedded(STORE.to_vec()),
    )
    .unwrap();
    for (alg, len) in [
        (Algorithm::Sha256, 32),
        (Algorithm::Sha384, 48),
        (Algorithm::Sha512, 64),
    ] {
        let binding = compute_data_hash(&embedded, alg, &Sha2).unwrap();
        assert_eq!(binding.hash.len(), len, "{alg:?}");
        assert!(
            verify_data_hash(&embedded, &binding, &Sha2).is_ok(),
            "{alg:?}"
        );
    }
}
