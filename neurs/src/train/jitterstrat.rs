/*!
 * An amorphous method of training a neural network.
 *
 * The method works by considering an example from the training set, and
 * testing the neural network on it multiple times, slightly 'jittering'(!) the
 * weights and biases of the network each time; after a certain desired number
 * of attempts, the current network weights are adjusted towards the best
 * performing variations.
 *
 * 'Amorphous' here means that the method itself could, in its general
 * form, apply to any set of parameters which can be measured with fitness.
 * In other words, it can be applied to a much more general case than neural
 * networks. However, the implementation provided here is specific to neural
 * networks, for the sake of performance and code simplicity.
 */
use crate::prelude::*;

use rand::thread_rng;
use rand_distr::*;

// Waiting for trait aliases to become stable so I can do this.
//    pub trait AJW = Fn(f32, f32, f32) -> f32;

/**
 * The weight-jitter training strategy.
 */
#[derive(Clone)]
pub struct WeightJitterStrat<AJW>
where
    AJW: Fn(f32, f32, f32) -> f32,
{
    /// How many different 'jitters' of the same weight should be tried.
    pub num_jitters: usize,

    /// Whether bad jitters should be taken into account when adjusting the
    /// current network's weights (by "moving away from" them).
    pub apply_bad_jitters: bool,

    /// An optional jitter width multiplier whose input is current best fitness.
    pub adaptive_jitter_width: Option<AJW>,

    /// How much the weights should be randomized in a jitter.
    pub jitter_width: f32,

    /// The amount of jitter_width that should be culled away with each epoch.
    pub jitter_width_falloff: f32,

    /// How much the weights should be adjusted after an epoch.
    pub step_factor: f32,

    /// How many cycles of compute and get-fitness should be run per network,
    /// per epoch.
    pub num_steps_per_epoch: usize,

    /* Internals. */
    pub curr_jitter_width: f32,
}

pub struct WeightJitterStratOptions<AJW>
where
    AJW: Fn(f32, f32, f32) -> f32,
{
    /// How many different 'jitters' of the same weight should be tried.
    pub num_jitters: usize,

    /// Whether bad jitters should be taken into account when adjusting the
    /// current network's weights (by "moving away from" them).
    pub apply_bad_jitters: bool,

    /// An optional jitter width multiplier whose input is current best fitness.
    pub adaptive_jitter_width: Option<AJW>,

    /// How much the weights should be randomized in a jitter.
    pub jitter_width: f32,

    /// The amount of jitter_width that should be culled away with each epoch.
    pub jitter_width_falloff: f32,

    /// How much the weights should be adjusted after an epoch.
    pub step_factor: f32,

    /// How many cycles of compute and get-fitness should be run per network,
    /// per epoch.
    pub num_steps_per_epoch: usize,
}

impl<AJW> WeightJitterStrat<AJW>
where
    AJW: Fn(f32, f32, f32) -> f32,
{
    pub fn new(options: WeightJitterStratOptions<AJW>) -> WeightJitterStrat<AJW> {
        WeightJitterStrat {
            num_jitters: options.num_jitters,
            jitter_width: options.jitter_width,
            jitter_width_falloff: options.jitter_width_falloff,
            step_factor: options.step_factor,
            adaptive_jitter_width: options.adaptive_jitter_width,
            num_steps_per_epoch: options.num_steps_per_epoch,
            apply_bad_jitters: options.apply_bad_jitters,

            curr_jitter_width: options.jitter_width,
        }
    }

    fn get_reference<AssemblyType, FrameType, H1, H2>(
        &mut self,
        assembly: &mut AssemblyType,
        frame: &mut FrameType,
    ) -> Result<(AssemblyWnb, f32), String>
    where
        AssemblyType: Assembly + Clone,
        FrameType: Frame<AssemblyType, ProdHandle = H1, TrainHandle = H2>,
        H1: FrameHandle<AssemblyType>,
        H2: FrameHandle<AssemblyType>,
    {
        let mut reference = frame
            .start_train_run(assembly.clone())
            .map_err(|(_, error_string)| error_string)?;

        while !reference.poll_state().is_done() {}

        if let FrameRunState::Error(err) = reference.poll_state() {
            return Err(err);
        }

        let reference_wnb = AssemblyWnb::from(&*assembly);
        Ok((reference_wnb, reference.get_fitness()))
    }
}

fn jitter_values<D: Distribution<f32>>(values: &mut [f32], distrib: D) {
    for value in values {
        *value += distrib.sample(&mut thread_rng());
    }
}

#[derive(Clone)]
struct WeightsAndBiases {
    w: Vec<f32>,
    b: Vec<f32>,
}

#[allow(unused)]
impl WeightsAndBiases {
    fn zero(&mut self) {
        self.w.fill(0.0);
        self.b.fill(0.0);
    }

    fn jitter<D: Distribution<f32>>(&mut self, distrib: &D) {
        jitter_values(&mut self.w, &distrib);
        jitter_values(&mut self.b, &distrib);
    }

    fn apply_to(&self, dest_layer: &mut NeuralLayer) {
        if cfg!(dbg) {
            assert!(dest_layer.weights.len() == self.w.len());
            assert!(dest_layer.biases.len() == self.b.len());
        }

        dest_layer.weights.clone_from(&self.w);
        dest_layer.biases.clone_from(&self.b);
    }

    fn scale(&mut self, scale: f32) {
        for w in &mut self.w {
            *w *= scale;
        }

        for b in &mut self.b {
            *b *= scale;
        }
    }

    fn scale_from(&mut self, other: &WeightsAndBiases, scale: f32) {
        for (i, ow) in other.w.iter().enumerate() {
            let w = self.w[i];
            let diff = w - ow;

            self.w[i] += diff * scale - diff;
        }

        for (i, ob) in other.b.iter().enumerate() {
            let b = self.b[i];
            let diff = b - ob;

            self.b[i] += diff * scale - diff;
        }
    }

    fn sub_from(&mut self, other: &WeightsAndBiases) {
        for (i, ow) in other.w.iter().enumerate() {
            self.w[i] -= ow;
        }

        for (i, ob) in other.b.iter().enumerate() {
            self.b[i] -= ob;
        }
    }

    fn add_to(&self, other: &mut WeightsAndBiases) {
        for (i, w) in self.w.iter().enumerate() {
            other.w[i] += w;
        }

        for (i, b) in self.b.iter().enumerate() {
            other.b[i] += b;
        }
    }
}

impl From<&NeuralLayer> for WeightsAndBiases {
    fn from(src_layer: &NeuralLayer) -> WeightsAndBiases {
        WeightsAndBiases {
            w: src_layer.weights.clone(),
            b: src_layer.biases.clone(),
        }
    }
}

impl From<&mut NeuralLayer> for WeightsAndBiases {
    fn from(src_layer: &mut NeuralLayer) -> WeightsAndBiases {
        WeightsAndBiases {
            w: src_layer.weights.clone(),
            b: src_layer.biases.clone(),
        }
    }
}

#[derive(Clone)]
struct NetworkWnb {
    wnbs: Vec<WeightsAndBiases>,
}

#[allow(unused)]
impl NetworkWnb {
    fn zero(&mut self) {
        for wnb in &mut self.wnbs {
            wnb.zero()
        }
    }

    fn apply_to(&self, dest_net: &mut SimpleNeuralNetwork) {
        if cfg!(dbg) {
            assert!(dest_net.layers.len() == self.wnbs.len());
        }

        for (i, wnb) in self.wnbs.iter().enumerate() {
            wnb.apply_to(&mut dest_net.layers[i]);
        }
    }

    fn jitter<D: Distribution<f32>>(&mut self, distrib: &D) {
        for wnb in &mut self.wnbs {
            wnb.jitter(&distrib);
        }
    }

    fn scale(&mut self, scale: f32) {
        for wnb in &mut self.wnbs {
            wnb.scale(scale);
        }
    }

    fn scale_from(&mut self, other: &NetworkWnb, scale: f32) {
        for (wnb, ownb) in self.wnbs.iter_mut().zip(&other.wnbs) {
            wnb.scale_from(ownb, scale);
        }
    }

    fn add_to(&self, other: &mut NetworkWnb) {
        for (wnb, ownb) in self.wnbs.iter().zip(&mut other.wnbs) {
            wnb.add_to(ownb);
        }
    }

    fn sub_from(&mut self, other: &NetworkWnb) {
        for (wnb, ownb) in self.wnbs.iter_mut().zip(&other.wnbs) {
            wnb.sub_from(ownb);
        }
    }
}

#[derive(Clone)]
struct AssemblyWnb {
    wnbs: Vec<NetworkWnb>,
}

#[allow(unused)]
impl AssemblyWnb {
    fn zero(&mut self) {
        for wnb in &mut self.wnbs {
            wnb.zero()
        }
    }

    fn apply_to<AS>(&self, dest_net: &mut AS)
    where
        AS: Assembly,
    {
        let mut netrefs = dest_net.get_networks_mut();

        for (nr, wnb) in netrefs.iter_mut().zip(self.wnbs.iter()) {
            wnb.apply_to(*nr);
        }
    }

    fn jitter<D: Distribution<f32>>(&mut self, distrib: &D) {
        for wnb in &mut self.wnbs {
            wnb.jitter(&distrib);
        }
    }

    fn scale(&mut self, scale: f32) {
        for wnb in &mut self.wnbs {
            wnb.scale(scale);
        }
    }

    fn scale_from(&mut self, other: &AssemblyWnb, scale: f32) {
        for (wnb, ownb) in self.wnbs.iter_mut().zip(&other.wnbs) {
            wnb.scale_from(ownb, scale);
        }
    }

    fn add_to(&self, other: &mut AssemblyWnb) {
        for (wnb, ownb) in self.wnbs.iter().zip(&mut other.wnbs) {
            wnb.add_to(ownb);
        }
    }

    fn sub_from(&mut self, other: &AssemblyWnb) {
        for (wnb, ownb) in self.wnbs.iter_mut().zip(&other.wnbs) {
            wnb.sub_from(ownb);
        }
    }
}

impl From<&SimpleNeuralNetwork> for NetworkWnb {
    fn from(src_net: &SimpleNeuralNetwork) -> NetworkWnb {
        NetworkWnb {
            wnbs: src_net.layers.iter().map(WeightsAndBiases::from).collect(),
        }
    }
}

impl From<&mut SimpleNeuralNetwork> for NetworkWnb {
    fn from(src_net: &mut SimpleNeuralNetwork) -> NetworkWnb {
        NetworkWnb {
            wnbs: src_net.layers.iter().map(WeightsAndBiases::from).collect(),
        }
    }
}

impl<AS> From<&AS> for AssemblyWnb
where
    AS: Assembly,
{
    fn from(src_as: &AS) -> AssemblyWnb {
        AssemblyWnb {
            wnbs: src_as
                .get_network_refs()
                .into_iter()
                .map(NetworkWnb::from)
                .collect(),
        }
    }
}

enum EpochJitterState<AssemblyType, HandleType>
where
    AssemblyType: Assembly + Clone,
    HandleType: FrameHandle<AssemblyType>,
{
    Pending(AssemblyType),
    Waiting(HandleType),
    Running(HandleType),
    Done(AssemblyType, f32),
    Error(AssemblyType, String),
}

struct EpochState<AssemblyType, HandleType>
where
    AssemblyType: Assembly + Clone,
    HandleType: FrameHandle<AssemblyType>,
{
    jitters: Vec<EpochJitterState<AssemblyType, HandleType>>,
}

impl<AssemblyType, HandleType> EpochJitterState<AssemblyType, HandleType>
where
    AssemblyType: Assembly + Clone,
    HandleType: FrameHandle<AssemblyType>,
{
    pub fn is_done(&self) -> bool {
        matches!(self, Done) || matches!(self, Error)
    }
}

impl<AssemblyType, HandleType> EpochState<AssemblyType, HandleType>
where
    AssemblyType: Assembly + Clone,
    HandleType: FrameHandle<AssemblyType>,
{
    pub fn init(template: &AssemblyType, num_jitters: usize, curr_jitter_width: f32) -> Self {
        EpochState {
            jitters: {
                let mut res = vec![];

                let reference_wnb: AssemblyWnb = AssemblyWnb::from(&*template);
                let distrib = Normal::<f32>::new(0.0, curr_jitter_width).unwrap();

                for _ in 0..num_jitters {
                    let mut net = template.clone();

                    let mut new_wnb: AssemblyWnb = reference_wnb.clone();

                    new_wnb.jitter(&distrib);
                    new_wnb.apply_to(&mut net);

                    res.push(EpochJitterState::Pending(net));
                }

                res
            },
        }
    }

    fn handle_to_state(mut handle: HandleType) -> EpochJitterState<AssemblyType, HandleType> {
        use EpochJitterState::*;

        let state = handle.poll_state();
        match state {
            FrameRunState::Waiting => Waiting(handle),
            FrameRunState::Running => Running(handle),
            FrameRunState::Done => {
                let fit = handle.get_fitness();
                Done(handle.finish(), fit)
            }
            FrameRunState::Error(str) => Error(handle.finish(), str),
        }
    }

    pub fn poll<FrameType, H1>(&mut self, frame: &mut FrameType) -> bool
    where
        FrameType: Frame<AssemblyType, ProdHandle = H1, TrainHandle = HandleType>,
    {
        for (index, state) in self.jitters.iter_mut().enumerate() {
            use EpochJitterState::*;
            match state {
                Pending(assembly) => {
                    if frame.can_run() {
                        self.jitters[index] = match frame.start_train_run(assembly) {
                            Ok(handle) => Self::handle_to_state(handle),
                            Err((assembly, str)) => EpochJitterState::Error(assembly, str),
                        }
                    }
                }

                Waiting(handle) | Running(handle) => {
                    self.jitters[index] = Self::handle_to_state(handle);
                }

                _ => {}
            }
        }

        self.all_done()
    }

    fn all_done(&self) -> bool {
        self.jitters.iter().all(|state| state.is_done())
    }

    pub fn results<FrameType, H1>(
        self,
        frame: &mut FrameType,
    ) -> Vec<Result<(AssemblyWnb, f32), String>>
    where
        FrameType: Frame<AssemblyType, ProdHandle = H1, TrainHandle = HandleType>,
    {
        self.jitters
            .into_iter()
            .map(|x| match x {
                EpochJitterState::Done(assembly, fit) => Ok((AssemblyWnb::from(&assembly), fit)),
                EpochJitterState::Error(assembly, err) => Err(err),
                _ => unreachable!(),
            })
            .collect()
    }

    pub fn results_ok_only<FrameType, H1>(self, frame: &mut FrameType) -> Vec<(AssemblyType, f32)>
    where
        FrameType: Frame<AssemblyType, ProdHandle = H1, TrainHandle = HandleType>,
    {
        self.jitters
            .into_iter()
            .filter_map(|x| match x {
                EpochJitterState::Done(assembly, fit) => Some((assembly, fit)),
                EpochJitterState::Error(assembly, err) => None,
                _ => unreachable!(),
            })
            .collect()
    }
}

impl<AJW> TrainingStrategy for WeightJitterStrat<AJW>
where
    AJW: Fn(f32, f32, f32) -> f32,
{
    fn reset_training(&mut self) {
        self.curr_jitter_width = self.jitter_width;
    }

    fn epoch<AssemblyType, FrameType, H1, H2>(
        &mut self,
        assembly: &mut AssemblyType,
        frame: &mut FrameType,
    ) -> Result<f32, String>
    where
        AssemblyType: Assembly + Clone,
        FrameType: Frame<AssemblyType, ProdHandle = H1, TrainHandle = H2>,
        H1: FrameHandle<AssemblyType>,
        H2: FrameHandle<AssemblyType>,
    {
        debug_assert!(self.num_jitters > 0);
        debug_assert!(self.jitter_width >= 0.0);
        debug_assert!(self.num_steps_per_epoch > 0);
        debug_assert!(self.step_factor >= 0.0);

        let (reference_wnb, reference_fitness) = self.get_reference(assembly, frame)?;

        let mut state: EpochState<AssemblyType, H2> =
            EpochState::init(assembly, self.num_jitters, self.curr_jitter_width);

        let results = state.results(frame);

        let results = results
            .into_iter()
            .filter_map(|x| x.ok())
            .collect::<Vec<_>>();

        let min_fitness = results
            .iter()
            .map(|x| x.1)
            .reduce(|ac, n| if ac < n { ac } else { n })
            .unwrap();
        let max_fitness = results
            .iter()
            .map(|x| x.1)
            .reduce(|ac, n| if ac > n { ac } else { n })
            .unwrap();

        let num_ok_jitters = if self.apply_bad_jitters {
            self.num_jitters
        } else {
            results
                .iter()
                .map(|x| if x.1 > 0.0 { 1_usize } else { 0_usize })
                .sum::<usize>()
        };

        let mut new_wnb: AssemblyWnb = reference_wnb.clone();

        if num_ok_jitters > 0 {
            let step_factor = self.step_factor / num_ok_jitters as f32;

            // Normalize delta fitnesses and use them to weight jitter weights
            // and biases proportionately when applying them to the ref. net.
            for (wnbs, fitness) in &mut results {
                if self.apply_bad_jitters || *fitness > 0.0 {
                    let fitness_scale = (*fitness - min_fitness)
                        / if max_fitness == min_fitness {
                            1.0
                        } else {
                            max_fitness - min_fitness
                        }
                        * 2.0
                        - 1.0;

                    wnbs.sub_from(&reference_wnb);
                    wnbs.scale((fitness_scale * step_factor) as f32);
                    wnbs.add_to(&mut new_wnb);
                }
            }

            //println!("Applied {} jitters.", num_ok_jitters);
        } else {

            //println!("Applied NO jitters.");
        }

        self.curr_jitter_width *= 1.0 - self.jitter_width_falloff;

        if self.adaptive_jitter_width.is_some() {
            self.curr_jitter_width = self.adaptive_jitter_width.as_ref().unwrap()(
                self.curr_jitter_width,
                (max_fitness - reference_fitness) as f32,
                (reference_fitness) as f32,
            );
        }

        new_wnb.apply_to(assembly);

        Ok(max_fitness + reference_fitness)
    }
}
