use crate::format::Format;
use std::fmt;

/// Errors returned by the crate.
#[derive(Debug)]
pub enum Error {
    /// The bytes did not match any supported ML model container format.
    UnknownFormat,
    /// The container was structurally malformed for its detected format.
    Malformed(String),
    /// No C2PA manifest was present in the model.
    NotFound,
    /// A manifest source carried neither an embedded store nor a remote URI.
    EmptySource,
    /// A stored manifest reference could not be decoded (e.g. invalid Base64).
    MalformedReference(String),
    /// More than one `c2pa:manifest` entry was present.
    ///
    /// Only one shall be present per file. The specification assigns a distinct
    /// failure code per format; see [`Error::code`].
    MultipleManifests(Format),
    /// The exclusion ranges are malformed: out of order, overlapping, extending
    /// past the end of the file, or not matching the located manifest value.
    ///
    /// For SafeTensors this also covers an 8-byte header length field that
    /// disagrees with the actual JSON header length, which the specification
    /// requires a validator to reject with `assertion.dataHash.malformed`.
    MalformedExclusion(String),
    /// The recomputed data hash did not match the value in the assertion.
    HashMismatch,
    /// A hash algorithm identifier outside the C2PA allowed list was requested.
    UnsupportedAlgorithm(String),
    Io(std::io::Error),
}

impl Error {
    /// The registered C2PA validation status code for this error, or `None`
    /// when the condition carries no status code.
    ///
    /// Reading, embedding, and format-detection errors are not validation
    /// outcomes and return `None`.
    pub fn code(&self) -> Option<&'static str> {
        Some(match self {
            Self::MultipleManifests(Format::Onnx) => "manifest.onnx.multipleManifests",
            Self::MultipleManifests(Format::SafeTensors) => {
                "manifest.safetensors.multipleManifests"
            }
            // GGUF embedding is a crate extension with no specified codes.
            Self::MultipleManifests(Format::Gguf) => return None,
            Self::MalformedExclusion(_) => "assertion.dataHash.malformed",
            Self::HashMismatch => "assertion.dataHash.mismatch",
            Self::UnsupportedAlgorithm(_) => "algorithm.unsupported",
            Self::UnknownFormat
            | Self::Malformed(_)
            | Self::NotFound
            | Self::EmptySource
            | Self::MalformedReference(_)
            | Self::Io(_) => return None,
        })
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownFormat => {
                write!(f, "unrecognized ML model container format")
            }
            Self::Malformed(s) => write!(f, "malformed model container: {s}"),
            Self::NotFound => write!(f, "no C2PA manifest found in model"),
            Self::EmptySource => {
                write!(f, "manifest source has neither an embedded store nor a URI")
            }
            Self::MalformedReference(s) => write!(f, "malformed manifest reference: {s}"),
            Self::MultipleManifests(container) => {
                write!(
                    f,
                    "more than one c2pa:manifest entry in the {} container",
                    container.name()
                )
            }
            Self::MalformedExclusion(s) => write!(f, "data hash exclusion is malformed: {s}"),
            Self::HashMismatch => write!(f, "data hash does not match the model content"),
            Self::UnsupportedAlgorithm(a) => write!(f, "unsupported hash algorithm: {a}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all() -> Vec<Error> {
        vec![
            Error::UnknownFormat,
            Error::Malformed("x".into()),
            Error::NotFound,
            Error::EmptySource,
            Error::MalformedReference("x".into()),
            Error::MultipleManifests(Format::Gguf),
            Error::MultipleManifests(Format::Onnx),
            Error::MultipleManifests(Format::SafeTensors),
            Error::MalformedExclusion("x".into()),
            Error::HashMismatch,
            Error::UnsupportedAlgorithm("sha1".into()),
        ]
    }

    #[test]
    fn display_composes_into_a_sentence_for_every_variant() {
        for e in all() {
            let s = e.to_string();
            assert!(!s.is_empty(), "{e:?} rendered empty");
            assert!(!s.ends_with('.'), "{e:?} ends with a period: {s}");
            let first = s.chars().next().expect("checked non-empty above");
            assert!(!first.is_uppercase(), "{e:?} starts uppercase: {s}");
        }
    }

    #[test]
    fn multiplicity_codes_are_per_format_and_as_registered() {
        assert_eq!(
            Error::MultipleManifests(Format::Onnx).code(),
            Some("manifest.onnx.multipleManifests")
        );
        assert_eq!(
            Error::MultipleManifests(Format::SafeTensors).code(),
            Some("manifest.safetensors.multipleManifests")
        );
        // GGUF is not a specified container, so inventing a code for it would be
        // inventing specification.
        assert_eq!(Error::MultipleManifests(Format::Gguf).code(), None);
    }

    #[test]
    fn every_code_is_a_registered_identifier() {
        for e in all() {
            if let Some(code) = e.code() {
                assert!(
                    matches!(
                        code,
                        "manifest.onnx.multipleManifests"
                            | "manifest.safetensors.multipleManifests"
                            | "assertion.dataHash.malformed"
                            | "assertion.dataHash.mismatch"
                            | "algorithm.unsupported"
                    ),
                    "{e:?} reports an unregistered code: {code}"
                );
            }
        }
    }

    #[test]
    fn reading_and_embedding_errors_carry_no_code() {
        for e in [
            Error::UnknownFormat,
            Error::Malformed("x".into()),
            Error::NotFound,
            Error::EmptySource,
            Error::MalformedReference("x".into()),
        ] {
            assert_eq!(e.code(), None, "{e:?} must not report a status code");
        }
    }

    #[test]
    fn format_names_itself_in_the_message() {
        assert!(Error::MultipleManifests(Format::Onnx)
            .to_string()
            .contains("ONNX"));
        assert!(Error::MultipleManifests(Format::SafeTensors)
            .to_string()
            .contains("SafeTensors"));
    }
}
