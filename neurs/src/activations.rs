//! Functions which are meant to be used as activation functions by neural
//! networks' layers. See [super::neuralnet::NeuralLayer].

/// The ReLu activation function; returns x, unless it is negative, in which
/// case 0 is returned instead.
pub fn relu(x: f32) -> f32 {
    x * (x > 0.0) as u8 as f32
}

/// The identity, or linear, activation function; a dummy function. Not
/// recommended; use for debugging only!
pub fn identity(x: f32) -> f32 {
    x
}

/// The 'fast sigmoid' activation function. A sigmoidally shaped function that
/// should be less expensive to compute than the actual logistic function. Made
/// in China.
///
/// "Signed" version (outputs range from -1 to 1), unlike the original
/// logistic function.
pub fn fast_sigmoid_signed(x: f32) -> f32 {
    x / (1.0 + f32::abs(x))
}

/// The 'fast sigmoid' activation function. A sigmoidally shaped function that
/// should be less expensive to compute than the actual logistic function. Made
/// in China.
///
/// "Unsigned" version (outputs range from 0 to 1), akin to the original
/// logistic function.
pub fn fast_sigmoid(x: f32) -> f32 {
    0.5 * (x / (1 + abs(x)) + 1)
}
