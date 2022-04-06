/*!
 * Label-based supervised learning frame for the TrainingFrame interface.
 */
use super::interface;

/// A label that can be used by the [LabeledLearningFrame].
pub trait TrainingLabel: Eq + Clone {
    /// How many label values there are under this label type.
    fn num_labels() -> u16;

    /// Turns a label into a one-hot index.
    fn index(&self) -> u16;

    /// Turns an one-hot index into a label.
    fn from_index(idx: u16) -> Self;

    /// A human-readable debug name, for debug reasons.
    fn debug_name(&self) -> String;
}

impl TrainingLabel for u16 {
    /// The 'index' of this label.
    ///
    /// This is important, because when autoencoding,
    /// a label is encoded as an one-hot vector (one where
    /// everything is 0, except for a given index, which becomes 1).
    fn index(self: &u16) -> u16 {
        *self
    }

    /// Converts from an index into a typed label.
    fn from_index(idx: u16) -> u16 {
        idx
    }

    /// The number of labels represented by this type.
    fn num_labels() -> u16 {
        u16::MAX
    }

    fn debug_name(&self) -> String {
        self.to_string()
    }
}

type DistanceWrapper = fn(f32) -> f32;

/**
 * A TrainingFrame implementation which simulates supervised learning
 * through labels.
 */
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

    curr_case: i16,

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
        if cfg!(debug) && cases_inputs.len() != cases_labels.len() {
            return Err("".to_owned());
        }

        Ok(Self {
            inputs: cases_inputs
                .iter()
                .cloned()
                .zip(cases_labels.iter().cloned())
                .collect(),

            curr_case: -1,

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
}

impl<T> interface::TrainingFrame for LabeledLearningFrame<T>
where
    T: TrainingLabel,
{
    fn next_training_case(&mut self) -> Vec<f32> {
        self.curr_case = ((self.curr_case + 1) as u16 % (self.inputs.len() as u16)) as i16;

        self.inputs[self.curr_case as usize].0.clone()
    }

    fn get_fitness(&mut self, inputs: &[f32], outputs: &[f32]) -> f32 {
        let desired_label = self.find_label_for(inputs).unwrap();
        let desired_idx = desired_label.index() as usize;

        -(outputs
            .iter()
            .enumerate()
            .map(|iout| {
                let (i, out) = iout;
                (self.distance_wrapper)(out - (if i == desired_idx { 1.0 } else { 0.0 }))
            })
            .sum::<f32>()
            / outputs.len() as f32)
    }

    fn get_reference_fitness(&mut self, inputs: &[f32], outputs: &[f32]) -> f32 {
        self.get_fitness(inputs, outputs)
    }
}
