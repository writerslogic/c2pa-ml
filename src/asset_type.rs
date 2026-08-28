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
    /// `c2pa.types.model.caffe` — a Caffe model.
    Caffe,
    /// `c2pa.types.model.caffe2` — a Caffe2 model.
    Caffe2,
    /// `c2pa.types.model.catboost` — a CatBoost model.
    CatBoost,
    /// `c2pa.types.model.coreml` — an Apple Core ML model.
    CoreMl,
    /// `c2pa.types.model.flax` — a Flax model.
    Flax,
    /// `c2pa.types.model.huggingface.transformers` — a Transformers model.
    HuggingFaceTransformers,
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
    /// `c2pa.types.model.lightgbm` — a LightGBM model.
    LightGbm,
    /// `c2pa.types.model.ml_net` — an ML.NET model.
    MlNet,
    /// `c2pa.types.model.mxnet` — an MXNet model.
    MxNet,
    /// `c2pa.types.model.openvino` — an OpenVINO model.
    OpenVino,
    /// `c2pa.types.model.openvino.parameter` — OpenVINO parameters.
    OpenVinoParameter,
    /// `c2pa.types.model.openvino.topology` — an OpenVINO topology.
    OpenVinoTopology,
    /// `c2pa.types.model.paddle` — a Paddle model.
    Paddle,
    /// `c2pa.types.model.sklearn` — a scikit-learn model.
    ScikitLearn,
    /// `c2pa.types.model.tensorrt` — an NVIDIA TensorRT model.
    TensorRt,
    /// `c2pa.types.model.tflite` — a TensorFlow Lite model.
    TfLite,
    /// `c2pa.types.model.torchscript` — a TorchScript model.
    TorchScript,
    /// `c2pa.types.model.xgboost` — an XGBoost model.
    XgBoost,
}

impl ModelType {
    /// The specification string for this model type.
    pub fn as_str(self) -> &'static str {
        match self {
            ModelType::Generic => "c2pa.types.model",
            ModelType::Caffe => "c2pa.types.model.caffe",
            ModelType::Caffe2 => "c2pa.types.model.caffe2",
            ModelType::CatBoost => "c2pa.types.model.catboost",
            ModelType::CoreMl => "c2pa.types.model.coreml",
            ModelType::Flax => "c2pa.types.model.flax",
            ModelType::HuggingFaceTransformers => "c2pa.types.model.huggingface.transformers",
            ModelType::Onnx => "c2pa.types.model.onnx",
            ModelType::PyTorch => "c2pa.types.model.pytorch",
            ModelType::TensorFlow => "c2pa.types.model.tensorflow",
            ModelType::Jax => "c2pa.types.model.jax",
            ModelType::Keras => "c2pa.types.model.keras",
            ModelType::LightGbm => "c2pa.types.model.lightgbm",
            ModelType::MlNet => "c2pa.types.model.ml_net",
            ModelType::MxNet => "c2pa.types.model.mxnet",
            ModelType::OpenVino => "c2pa.types.model.openvino",
            ModelType::OpenVinoParameter => "c2pa.types.model.openvino.parameter",
            ModelType::OpenVinoTopology => "c2pa.types.model.openvino.topology",
            ModelType::Paddle => "c2pa.types.model.paddle",
            ModelType::ScikitLearn => "c2pa.types.model.sklearn",
            ModelType::TensorRt => "c2pa.types.model.tensorrt",
            ModelType::TfLite => "c2pa.types.model.tflite",
            ModelType::TorchScript => "c2pa.types.model.torchscript",
            ModelType::XgBoost => "c2pa.types.model.xgboost",
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
