/**
 * An interface that any neural network training method should support.
 *
 * A training method is actually an implementation of [TrainingStrategy].
 */
use super::super::neuralnet::SimpleNeuralNetwork;

/**
 * A set of rules for measuring a network's fitness, and provides inputs and
 * outputs to train on.
 *
 * Expected to be stateful... and tasteful.
 */
pub trait TrainingFrame {
    /**
     * Return the next set of inputs that the next training epoch should be
     * about.
     */
    fn next_training_case(&mut self) -> Vec<f32>;

    /**
     * How well the network scores by providing an output to an input suggested
     * by the training frame.
     */
    fn get_fitness(&mut self, inputs: &[f32], outputs: &[f32]) -> f32;

    /**
     * Gets the reference fitness fot a network, at the beginning of its training.
     */
    fn get_reference_fitness(&mut self, inputs: &[f32], outputs: &[f32]) -> f32;
}

/**
 * The particular strategy a [Trainer] can employ to adjust the
 * weights of a neural network according to the training inputs and fitness
 * score provided by the [TrainingFrame].
 */
pub trait TrainingStrategy {
    /**
     * Perform an epoch of training on the neural network.
     *
     * Should return the best fitness arising from this epoch.
     */
    fn epoch(
        &mut self,
        net: &mut SimpleNeuralNetwork,
        frame: &mut Box<dyn TrainingFrame>,
    ) -> Result<f32, String>;
}
