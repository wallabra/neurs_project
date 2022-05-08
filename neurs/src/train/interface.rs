/*!
 * An interface that any neural network training method should support.
 *
 * A training method is actually an implementation of [TrainingStrategy].
 */
use async_trait::async_trait;

use crate::prelude::{Assembly, AssemblyFrame};

#[async_trait]
/**
 * The particular strategy a [super::trainer::Trainer] can employ to adjust the
 * weights of a neural network according to the training inputs and fitness
 * score.
 */
pub trait TrainingStrategy<AssemblyType, ATF>
where
    AssemblyType: Assembly,
    ATF: AssemblyFrame<AssemblyType>,
{
    /**
     * Reset the TrainingStrategy's internals for a new training session.
     */
    fn reset_training(&mut self);

    /**
     * Perform an epoch of training on the neural network.
     *
     * Should return a promise of the best fitness arising from this epoch.
     */
    async fn epoch(
        &mut self,
        assembly: &mut AssemblyType,
        assembly_frame: &mut ATF,
    ) -> Result<f64, String>;
}
