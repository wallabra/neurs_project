//! Code for the assembly of multiple networks.

use async_trait::async_trait;

use crate::prelude::SimpleNeuralNetwork;

#[cfg(feature = "serde")]
#[macro_use]
extern crate serde;

/// An assembly; an use case where multiple networks are required for
/// something.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub trait Assembly {
    /// Get immutable references to the neural networks used by this
    /// assembly.
    fn get_network_refs(&self) -> &[SimpleNeuralNetwork];

    /// Get mutable references to the neural networks used by this
    /// assembly.
    fn get_networks_mut(&mut self) -> &mut [SimpleNeuralNetwork];

    /// Get the number of networks in this Assembly.
    fn len(&self) -> usize {
        self.get_network_refs().len()
    }

    /// Whether this assembly is empty.
    ///
    /// Provided to satisfy cargo clippy; expect the result to always be false.
    fn is_empty(&self) -> bool {
        self.get_network_refs().is_empty()
    }
}

impl Assembly for SimpleNeuralNetwork {
    fn get_network_refs(&self) -> &[SimpleNeuralNetwork] {
        std::slice::from_ref(self)
    }

    fn get_networks_mut(&mut self) -> &mut [SimpleNeuralNetwork] {
        std::slice::from_mut(self)
    }

    fn len(&self) -> usize {
        1
    }

    fn is_empty(&self) -> bool {
        false
    }
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
