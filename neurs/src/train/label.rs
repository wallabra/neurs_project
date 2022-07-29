/*!
 * Label-based supervised learning frame for the TrainingFrame interface.
 */
use crate::prelude::*;

use async_trait::async_trait;

/// A label that can be used by the [LabeledLearningFrame].
pub trait TrainingLabel: Eq + Clone + Send {
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

impl TrainingLabel for bool {
    /// The 'index' of this label.
    ///
    /// This is important, because when autoencoding,
    /// a label is encoded as an one-hot vector (one where
    /// everything is 0, except for a given index, which becomes 1).
    fn index(self: &bool) -> usize {
        if *self {
            1
        } else {
            0
        }
    }

    /// Converts from an index into a typed label.
    fn from_index(idx: usize) -> bool {
        idx > 0
    }

    /// The number of labels represented by this type.
    fn num_labels() -> usize {
        2
    }

    /// The human-readable name of a label for debugging.
    fn debug_name(&self) -> String {
        self.to_string()
    }
}

type DistanceWrapper = fn(f64) -> f64;

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

    /// The metric to use to measure the error of an output.
    ///
    /// Used when verifying whether the one-hot encoded output of a network in
    /// a training case matches the expected output as per the case's
    /// corresponding label.
    distance_wrapper: Box<DistanceWrapper>,
}

impl<T> LabeledLearningFrame<T>
where
    T: TrainingLabel,
{
    pub fn new(
        cases_inputs: Vec<Vec<f32>>,
        cases_labels: Vec<T>,
        distance_wrapper: Option<Box<DistanceWrapper>>,
    ) -> Result<Self, String> {
        if (cfg!(debug) || cfg!(tests)) && cases_inputs.len() != cases_labels.len() {
            return Err("".to_owned());
        }

        Ok(Self {
            inputs: cases_inputs
                .iter()
                .cloned()
                .zip(cases_labels.iter().cloned())
                .collect(),

            distance_wrapper: Box::from(
                distance_wrapper.map_or(f64::abs as fn(f64) -> f64, |x| *x),
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

/// A classifier assembly.
pub struct NeuralClassifier {
    pub classifier: SimpleNeuralNetwork,
}

impl Assembly for NeuralClassifier {
    fn get_network_refs(&self) -> &[&SimpleNeuralNetwork] {
        &[&self.classifier]
    }

    fn get_networks_mut(&mut self) -> &[&mut SimpleNeuralNetwork] {
        &[&mut self.classifier]
    }
}

#[async_trait]
impl<T> AssemblyFrame<NeuralClassifier> for LabeledLearningFrame<T>
where
    T: TrainingLabel,
{
    type E = String;

    async fn run(&mut self, assembly: &mut NeuralClassifier) -> Result<f64, String> {
        let mut fitness = 0.0_f64;
        let mut outputs = vec![0.0_f32; T::num_labels()];

        for (case, desired_label) in &self.inputs {
            let desired_idx = desired_label.index() as usize;

            assembly.classifier.compute_values(&case, &mut outputs);

            fitness -= outputs
                .iter()
                .enumerate()
                .map(|iout| {
                    let (i, out) = iout;
                    (self.distance_wrapper)(
                        *out as f64 - (if i == desired_idx { 1.0 } else { 0.0 }),
                    )
                })
                .sum::<f64>()
                / outputs.len() as f64;
        }

        Ok(fitness)
    }
}

impl<LT> LabeledLearningFrame<LT>
where
    LT: TrainingLabel,
{
    pub fn avg_reference_fitness(
        &mut self,
        assembly: &mut NeuralClassifier,
    ) -> Result<f64, String> {
        let mut fitness = 0.0_f64;
        let mut outputs = vec![0.0_f32; LT::num_labels()];

        for (case, desired_label) in &self.inputs {
            let desired_idx = desired_label.index() as usize;

            assembly.classifier.compute_values(&case, &mut outputs);

            fitness -= outputs
                .iter()
                .enumerate()
                .map(|iout| {
                    let (i, out) = iout;
                    (self.distance_wrapper)(
                        *out as f64 - (if i == desired_idx { 1.0 } else { 0.0 }),
                    )
                })
                .sum::<f64>()
                / outputs.len() as f64;
        }

        Ok(fitness)
    }
}
