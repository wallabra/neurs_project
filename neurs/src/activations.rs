//! Functions which are meant to be used as activation functions by neural
//! networks' layers. See [super::neuralnet::NeuralLayer].

/// The ReLu activation function; returns x, unless it is negative, in which
/// case 0 is returned instead.
#[inline(always)]
pub fn relu(x: f32) -> f32 {
    x * ((x > 0.0) as u8 as f32)
}

/// The identity, or linear, activation function; a dummy function. Not
/// recommended; use for debugging only!
#[inline(always)]
pub fn identity(x: f32) -> f32 {
    x
}

/// The 'fast sigmoid' activation function. A sigmoidally shaped function that
/// should be less expensive to compute than the actual logistic function. Made
/// in China.
///
/// "Signed" version (outputs range from -1 to 1), unlike the original
/// logistic function.
#[inline(always)]
pub fn fast_sigmoid_signed(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

/// The 'fast sigmoid' activation function. A sigmoidally shaped function that
/// should be less expensive to compute than the actual logistic function. Made
/// in China.
///
/// "Unsigned" version (outputs range from 0 to 1), akin to the original
/// logistic function.
#[inline(always)]
pub fn fast_sigmoid(x: f32) -> f32 {
    0.5 * (x / (1.0 + x.abs()) + 1.0)
}

/// The original sigmoid activation function.
///
/// If precision is not required, use [fast_sigmoid] or [fast_sigmoid_signed].
#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + -x.exp())
}

/// The SiLu (swish) function - x multiplied with its own sigmoid.
///
/// This uses the accurate [sigmoid] implementation from above. If performance is
/// more important, use [fast_silu].
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// The SiLu (swish) function - x multiplied with its own sigmoid.
///
/// This uses the quick and dirty [fast_sigmoid] implementatiom from above.
#[inline(always)]
pub fn fast_silu(x: f32) -> f32 {
    x * fast_sigmoid(x)
}

/// Softplus - a smoother version of ReLu.
#[inline(always)]
pub fn softplus(x: f32) -> f32 {
    (1 + x.exp()).log()
}
