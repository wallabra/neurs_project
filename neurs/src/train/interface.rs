/*!
 * An interface that any neural network training method should support.
 *
 * A training method is actually an implementation of [TrainingStrategy].
 */
use crate::prelude::*;

/**
 * The particular strategy a [super::trainer::Trainer] can employ to adjust the
 * weights of a neural network according to the training inputs and fitness
 * score.
 */
pub trait TrainingStrategy {
    /**
     * Reset the TrainingStrategy's internals for a new training session.
     */
    fn reset_training(&mut self);

    /**
     * Perform an epoch of training on the neural network.
     *
     * Should return a promise of the best fitness arising from this epoch.
     */
    fn epoch<AssemblyType, FrameType, H1, H2>(
        &mut self,
        assembly: &mut AssemblyType,
        assembly_frame: &mut FrameType,
    ) -> Result<f32, String>
    where
        AssemblyType: Assembly + Clone,
        FrameType: Frame<AssemblyType, ProdHandle = H1, TrainHandle = H2>,
        H1: FrameHandle<AssemblyType>,
        H2: FrameHandle<AssemblyType>;
}
