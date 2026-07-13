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
    Io(std::io::Error),
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
