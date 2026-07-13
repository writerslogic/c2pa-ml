//! The C2PA asset type values for AI/ML assets, from Table 11 ("Asset type
//! values") of the [C2PA Technical Specification].
//!
//! The core specification defines no dedicated *embedding* method for model
//! container formats; instead a C2PA Manifest embedded in a model uses the
//! [asset type assertion] (`c2pa.asset-type`) to declare what the asset is. This
//! module exposes the canonical `c2pa.types.model.*` strings so a claim
//! generator can populate that assertion consistently with how this crate
//! embedded the manifest.
//!
//! [C2PA Technical Specification]: https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html
//! [asset type assertion]: https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html#_asset_type

/// A C2PA model asset type (a value of the asset type assertion's `type` field).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    /// `c2pa.types.model` — a model not described by any other value.
    Generic,
    /// `c2pa.types.model.onnx` — an ONNX model.
    Onnx,
    /// `c2pa.types.model.pytorch` — a PyTorch model.
    PyTorch,
    /// `c2pa.types.model.tensorflow` — a TensorFlow model.
    TensorFlow,
    /// `c2pa.types.model.jax` — a JAX model.
    Jax,
    /// `c2pa.types.model.keras` — a Keras model.
    Keras,
    /// `c2pa.types.model.mxnet` — an MXNet model.
    MxNet,
    /// `c2pa.types.model.openvino` — an OpenVINO model.
    OpenVino,
}

impl ModelType {
    /// The specification string for this model type.
    pub fn as_str(self) -> &'static str {
        match self {
            ModelType::Generic => "c2pa.types.model",
            ModelType::Onnx => "c2pa.types.model.onnx",
            ModelType::PyTorch => "c2pa.types.model.pytorch",
            ModelType::TensorFlow => "c2pa.types.model.tensorflow",
            ModelType::Jax => "c2pa.types.model.jax",
            ModelType::Keras => "c2pa.types.model.keras",
            ModelType::MxNet => "c2pa.types.model.mxnet",
            ModelType::OpenVino => "c2pa.types.model.openvino",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_strings() {
        assert_eq!(ModelType::Generic.as_str(), "c2pa.types.model");
        assert_eq!(ModelType::Onnx.as_str(), "c2pa.types.model.onnx");
        assert_eq!(ModelType::PyTorch.as_str(), "c2pa.types.model.pytorch");
    }
}
