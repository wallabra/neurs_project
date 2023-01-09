/*!
 * Code for the Trainer, the orchestration structore of neural network
 * training.
 */
use crate::prelude::{Assembly, SimpleFrame, TrainingStrategy};

/**
 * A struct which orchestrates the training process of a neural network.
 *
 * Holds the state of training; a current network, a [SimpleFrame]
 * and a [TrainingStrategy].
 */
pub struct Trainer<'a, AssemblyType, ATF, TS>
where
    AssemblyType: Assembly + Send,
    ATF: SimpleFrame<AssemblyType>,
    TS: TrainingStrategy,
{
    /**
     * The current reference neural network of this trainer.
     */
    pub reference_assembly: &'a mut AssemblyType,

    /**
     * The current training frame of this trainer.
     */
    pub frame: ATF,

    /**
     * The current training strategy of this trainer.
     *
     * This is the particular method by which a network is trained.
     */
    pub strategy: TS,
}

impl<'a, AssemblyType, ATF, TS> Trainer<'a, AssemblyType, ATF, TS>
where
    AssemblyType: Assembly + Send,
    ATF: SimpleFrame<AssemblyType>,
    TS: TrainingStrategy,
{
    /**
     * Refer to an existing assembly, and make it this trainer's reference one.
     */
    pub fn new(
        assembly: &'a mut AssemblyType,
        frame: ATF,
        strategy: TS,
    ) -> Trainer<AssemblyType, ATF, TS> {
        Trainer {
            reference_assembly: assembly,
            frame,
            strategy,
        }
    }

    /**
     * Perform a single epoch of training.
     *
     * Should return the best fitness arising from this epoch.
     */
    pub fn epoch(&mut self) -> Result<f32, String> {
        self.strategy
            .epoch(self.reference_assembly, &mut self.frame)
    }
}
