/*!
 * Code for the Trainer, the orchestration structore of neural network
 * training.
 */
use super::super::neuralnet::SimpleNeuralNetwork;
use super::interface::{TrainingFrame, TrainingStrategy};

/**
 * A struct which orchestrates the training process of a neural network.
 *
 * Holds the state of training; a current network, a [TrainingFrame]
 * and a [TrainingStrategy].
 */
pub struct Trainer<'a> {
    /**
     * The current reference neural network of this trainer.
     */
    pub reference_net: &'a mut SimpleNeuralNetwork,

    /**
     * The current training frame of this trainer.
     */
    pub frame: Box<dyn TrainingFrame>,

    /**
     * The current training strategy of this trainer.
     *
     * This is the particular method by which a network is trained.
     */
    pub strategy: Box<dyn TrainingStrategy>,
}

impl<'a> Trainer<'a> {
    /**
     * Refer to an existing reference-counted neural network, and make it this
     * trainer's reference network.
     */
    pub fn new(
        network: &'a mut SimpleNeuralNetwork,
        frame: Box<dyn TrainingFrame>,
        strategy: Box<dyn TrainingStrategy>,
    ) -> Trainer {
        Trainer {
            reference_net: network,
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
        self.strategy.epoch(self.reference_net, &mut self.frame)
    }
}
