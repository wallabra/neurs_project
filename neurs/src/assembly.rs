//! Code for the assembly of multiple networks.

use crate::prelude::SimpleNeuralNetwork;

/// An assembly; an use case where multiple networks are required for
/// something.
pub trait Assembly {
    /// Get immutable references to the neural networks used by this
    /// assembly.
    fn get_network_refs(&self) -> &[&SimpleNeuralNetwork];

    /// Get mutable references to the neural networks used by this
    /// assembly.
    fn get_networks_mut(&mut self) -> &[&mut SimpleNeuralNetwork];
}

/// Parameters and specifics for how an Assembly is used and trained.
pub trait AssemblyFrame<AssemblyType>
where
    AssemblyType: Assembly,
{
    /// Performs a training run.
    /// Returns a promise of a fitness value.
    fn run(&mut self, assembly: &mut AssemblyType) -> Promise<f64, String>;
}
