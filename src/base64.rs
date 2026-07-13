//! Minimal standard Base64 (RFC 4648) encode/decode with no dependencies.
//!
//! Used to carry a binary C2PA Manifest Store inside the string-typed metadata
//! slots of SafeTensors (`__metadata__`) and ONNX (`metadata_props`).

const CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub(crate) fn encode(input: &[u8]) -> String {
    let mut out = Vec::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(CHARS[((triple >> 18) & 0x3F) as usize]);
        out.push(CHARS[((triple >> 12) & 0x3F) as usize]);
        out.push(if chunk.len() > 1 {
            CHARS[((triple >> 6) & 0x3F) as usize]
        } else {
            b'='
        });
        out.push(if chunk.len() > 2 {
            CHARS[(triple & 0x3F) as usize]
        } else {
            b'='
        });
    }
    // Safe: every pushed byte is ASCII.
    String::from_utf8(out).expect("base64 alphabet is ASCII")
}

pub(crate) fn decode(input: &str) -> Result<Vec<u8>, &'static str> {
    let s = input.trim();
    let bytes = s.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return Err("length is not a multiple of 4");
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for quad in bytes.chunks(4) {
        let mut acc = 0u32;
        let mut pad = 0;
        for (i, &c) in quad.iter().enumerate() {
            let v = match c {
                b'A'..=b'Z' => (c - b'A') as u32,
                b'a'..=b'z' => (c - b'a' + 26) as u32,
                b'0'..=b'9' => (c - b'0' + 52) as u32,
                b'+' => 62,
                b'/' => 63,
                b'=' if i >= 2 => {
                    pad += 1;
                    0
                }
                _ => return Err("invalid base64 character"),
            };
            acc = (acc << 6) | v;
        }
        out.push((acc >> 16) as u8);
        if pad < 2 {
            out.push((acc >> 8) as u8);
        }
        if pad < 1 {
            out.push(acc as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors() {
        assert_eq!(encode(b""), "");
        assert_eq!(encode(b"f"), "Zg==");
        assert_eq!(encode(b"fo"), "Zm8=");
        assert_eq!(encode(b"foo"), "Zm9v");
        assert_eq!(encode(b"foob"), "Zm9vYg==");
        assert_eq!(encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn round_trip() {
        for v in [
            &b""[..],
            b"f",
            b"fo",
            b"foo",
            b"foobar",
            &[0u8, 1, 2, 3, 255, 254, 253],
        ] {
            assert_eq!(decode(&encode(v)).unwrap(), v);
        }
    }

    #[test]
    fn rejects_malformed() {
        assert!(decode("Zg=").is_err());
        assert!(decode("Zg!=").is_err());
    }
}
