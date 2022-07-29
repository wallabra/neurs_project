//! Code for the assembly of multiple networks.

use async_trait::async_trait;

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
#[async_trait]
pub trait AssemblyFrame<AssemblyType>
where
    AssemblyType: Assembly,
{
    type E: ToString;

    /// Performs a training run.
    /// Returns a promise of a fitness value.
    async fn run(&mut self, assembly: &mut AssemblyType) -> Result<f32, Self::E>;
}
