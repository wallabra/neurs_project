/*!
 * Label-based supervised learning frame for the TrainingFrame interface.
 */
use super::{interface, TrainingFrame};
use crate::prelude::*;

/// A label that can be used by the [LabeledLearningFrame].
pub trait TrainingLabel: Eq + Clone {
    /// How many label values there are under this label type.
    fn num_labels() -> usize;

    /// Turns a label into a one-hot index.
    fn index(&self) -> usize;

    /// Turns an one-hot index into a label.
    fn from_index(idx: usize) -> Self;

    /// A human-readable debug name, for debug reasons.
    fn debug_name(&self) -> String;
}

impl TrainingLabel for usize {
    /// The 'index' of this label.
    ///
    /// This is important, because when autoencoding,
    /// a label is encoded as an one-hot vector (one where
    /// everything is 0, except for a given index, which becomes 1).
    fn index(self: &usize) -> usize {
        *self
    }

    /// Converts from an index into a typed label.
    fn from_index(idx: usize) -> usize {
        idx
    }

    /// The number of labels represented by this type.
    fn num_labels() -> usize {
        usize::MAX
    }

    /// The human-readable name of a label for debugging.
    fn debug_name(&self) -> String {
        self.to_string()
    }
}

type DistanceWrapper = fn(f32) -> f32;

/**
 * A TrainingFrame implementation which simulates supervised learning
 * through labels.
 */
#[derive(Clone)]
pub struct LabeledLearningFrame<LabelType>
where
    LabelType: TrainingLabel,
{
    /**
     * A list of pairs of inputs and associated labels.
     *
     * The network is supposed to eventually learn each input
     * to its associated label.
     */
    inputs: Vec<(Vec<f32>, LabelType)>,

    /// The currently used training case.
    curr_case: usize,

    /// The metric to use to measure the error of an output.
    ///
    /// Used when verifying whether the one-hot encoded output of a network in
    /// a training case matches the expected output as per the case's
    /// corresponding label.
    distance_wrapper: Box<DistanceWrapper>,

    /// Whether reference_fitness should be used.
    ///
    /// If this is false, get_reference_fitness will always return zero.
    use_reference_fitness: bool,
}

impl<T> LabeledLearningFrame<T>
where
    T: TrainingLabel,
{
    pub fn new(
        cases_inputs: Vec<Vec<f32>>,
        cases_labels: Vec<T>,
        distance_wrapper: Option<Box<DistanceWrapper>>,
        use_reference_fitness: bool,
    ) -> Result<Self, String> {
        if (cfg!(debug) || cfg!(tests)) && cases_inputs.len() != cases_labels.len() {
            return Err("".to_owned());
        }

        Ok(Self {
            use_reference_fitness,

            inputs: cases_inputs
                .iter()
                .cloned()
                .zip(cases_labels.iter().cloned())
                .collect(),

            curr_case: cases_inputs.len() - 1,

            distance_wrapper: Box::from(
                distance_wrapper.map_or(f32::abs as fn(f32) -> f32, |x| *x),
            ),
        })
    }

    fn find_label_for(&self, inputs: &[f32]) -> Option<&T> {
        for inp in &self.inputs {
            if inp.0 == inputs {
                return Some(&inp.1);
            }
        }

        None
    }

    /**
     * The number of training cases registered.
     *
     * Each network should be tested against all of them.
     */
    pub fn num_cases(&self) -> usize {
        self.inputs.len()
    }
}

impl<T> interface::TrainingFrame for LabeledLearningFrame<T>
where
    T: TrainingLabel,
{
    /**
     * Reset the training frame; called before each network is trained in the
     * training process.
     */
    fn reset_frame(&mut self) {
        self.curr_case = self.inputs.len() - 1;
    }

    fn next_training_case(&mut self) -> Vec<f32> {
        self.curr_case = (self.curr_case + 1) % self.inputs.len();

        self.inputs[self.curr_case as usize].0.clone()
    }

    fn get_fitness(&mut self, inputs: &[f32], outputs: &[f32]) -> f32 {
        let desired_label = self.find_label_for(inputs).unwrap();
        let desired_idx = desired_label.index() as usize;

        let fitness = -(outputs
            .iter()
            .enumerate()
            .map(|iout| {
                let (i, out) = iout;
                (self.distance_wrapper)(out - (if i == desired_idx { 1.0 } else { 0.0 }))
            })
            .sum::<f32>()
            / outputs.len() as f32);

        if cfg!(debug) || cfg!(tests) {
            assert!(fitness <= 0.0);
        }

        fitness
    }

    fn get_reference_fitness(&mut self, inputs: &[f32], outputs: &[f32]) -> f32 {
        if self.use_reference_fitness {
            self.get_fitness(inputs, outputs)
        } else {
            0.0
        }
    }
}

impl<LT> LabeledLearningFrame<LT>
where
    LT: TrainingLabel,
{
    pub fn avg_reference_fitness(&mut self, net: &mut SimpleNeuralNetwork) -> Result<f32, String> {
        let mut fit = 0.0;
        let mut output = vec![0.0; net.output_size()?];

        self.reset_frame();

        for _ in 0..self.inputs.len() {
            let reference_input = self.next_training_case();
            net.compute_values(&reference_input, &mut output)?;
            fit += self.get_reference_fitness(&reference_input, &output);
        }

        Ok(fit / self.inputs.len() as f32)
    }
}
