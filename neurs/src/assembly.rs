//! Code for the assembly of multiple networks.

use crate::prelude::SimpleNeuralNetwork;

/// An assembly; an use case where multiple networks are required for
/// something.
pub trait Assembly {
    /// Get immutable references to the neural networks used by this
    /// assembly.
    fn get_network_refs(&self) -> Vec<&SimpleNeuralNetwork>;

    /// Get mutable references to the neural networks used by this
    /// assembly.
    fn get_networks_mut(&mut self) -> Vec<&mut SimpleNeuralNetwork>;
}
