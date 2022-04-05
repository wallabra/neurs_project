/**
 * Code for the Trainer, the orchestration structore of neural network
 * training.
 */
use super::super::neuralnet::{NeuralLayer, SimpleNeuralNetwork};
use super::interface::{TrainingFrame, TrainingStrategy};

/**
 * A struct which orchestrates the training process of a neural network.
 *
 * Holds the state of training; a current network, a [TrainingFrame]
 * and a [TrainingStrategy].
 */
pub struct Trainer {
    /**
     * The current reference neural network of this trainer.
     */
    pub reference_net: SimpleNeuralNetwork,

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

impl Trainer {
    /**
     * Create a new Trainer with a randomly initialized neural network.
     *
     * This is pretty much starting from square one.
     *
     * Provide a function that can initialize the neural network's
     * structure to your liking; it takes no arguments, but should
     * return a list of layers, usually newly constructed ones (that,
     * as such, contain random weights and biases).
     */
    pub fn new_random(
        layer_initializer: fn() -> Vec<NeuralLayer>,
        frame: Box<dyn TrainingFrame>,
        strategy: Box<dyn TrainingStrategy>,
    ) -> Trainer {
        let mut new_network = SimpleNeuralNetwork { layers: vec![] };

        new_network.layers.append(&mut layer_initializer());

        Trainer {
            reference_net: new_network,
            frame,
            strategy,
        }
    }

    /**
     * Copy an existing neural network and make it this trainer's reference network.
     */
    pub fn new_from_net(
        network: &SimpleNeuralNetwork,
        frame: Box<dyn TrainingFrame>,
        strategy: Box<dyn TrainingStrategy>,
    ) -> Trainer {
        Trainer {
            reference_net: network.clone(),
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
            .epoch(&mut self.reference_net, &mut self.frame)
    }
}
