/*!
 * An interface that any neural network training method should support.
 *
 * A training method is actually an implementation of [TrainingStrategy].
 */
use super::super::assembly::{Assembly, AssemblyFrame};

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
     * Should return the best fitness arising from this epoch.
     */
    fn epoch(
        &mut self,
        assembly: &mut AssemblyType,
        assembly_frame: &mut ATF,
    ) -> Promise<f64, String>;
}
