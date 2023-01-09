/*!
 * The neural network and interface code.
 */
pub mod activations;
pub mod assembly;
pub mod neuralnet;
pub mod train;

pub mod prelude {
    /*!
     * A set of useful imports to always have.
     */
    pub use super::activations;
    pub use super::assembly::*;
    pub use super::neuralnet::*;
    pub use super::train::prelude::*;
}
