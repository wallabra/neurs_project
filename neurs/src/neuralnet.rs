/*!
 * A basic neural network structure.
 *
 * It is quite simplistic, only supporting
 * feed-forward neural networks of dense layers.
 * It also provides a default activation function,
 * the ReLu, although any can be supplied.
 */
use super::activations::relu;
use rand::prelude::*;
use rand_distr::*;

/**
 * A neural network activation function.
 */
pub type NNActivation = fn(f32) -> f32;

/**
 * A simple dense layer.
 */
#[derive(Clone)]
pub struct NeuralLayer {
    /// The activation function of the layer.
    pub activation: Box<NNActivation>,

    /// The weights of the layer.
    pub weights: Vec<f32>,

    /// The biases of the layer.
    pub biases: Vec<f32>,

    /// The input size of the layer.
    pub input_size: usize,

    /// The output size of the layer.
    pub output_size: usize,

    /// The product of the input and output sizes of the layer.
    pub area: u32,
}

impl NeuralLayer {
    /// Create a dense layer with random weights and biases, from an input and output
    /// sizes and an activation function.
    ///
    /// If `activation` is `None`, it will default to [relu].
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: Option<NNActivation>,
    ) -> NeuralLayer {
        let activation = activation.unwrap_or(relu);

        let area: u32 = input_size as u32 * output_size as u32;

        let mut weights: Vec<f32> = vec![0.0; area as usize];
        let mut biases: Vec<f32> = vec![0.0; output_size as usize];

        let mut random_distrib = Normal::<f32>::new(0.0, 1.0)
            .unwrap()
            .sample_iter(thread_rng());

        weights
            .as_mut_slice()
            .fill_with(|| random_distrib.next().unwrap());
        biases
            .as_mut_slice()
            .fill_with(|| random_distrib.next().unwrap());

        NeuralLayer {
            activation: Box::from(activation),

            weights,
            biases,

            input_size,
            output_size,
            area,
        }
    }

    /// Transforms a vector of values through this dense layer of neurons.
    pub fn compute(&self, mut inputs: &[f32], mut outputs: &mut [f32]) -> Result<(), String> {
        if cfg!(debug) || cfg!(tests) {
            if inputs.len() < self.input_size as usize {
                return Err("Source slice is smaller than the input size of this layer".to_owned());
            }

            if outputs.len() < self.output_size as usize {
                return Err(
                    "Destination slice is smaller than the output size of this layer".to_owned(),
                );
            }
        }

        inputs = &inputs[0..self.input_size];
        outputs = &mut outputs[0..self.output_size];

        for (i, out) in outputs.iter_mut().enumerate() {
            let idx_base: usize = (i * self.input_size) as usize;

            let value = (self.activation)(
                self.biases[i]
                    + inputs
                        .iter()
                        .zip(&self.weights[idx_base..])
                        .map(|(inp, w)| (*inp) * (*w))
                        .sum::<f32>(),
            );

            *out = value;
        }

        Ok(())
    }
}

/**
 * A simple feed-forward neural network.
 */
#[derive(Clone)]
pub struct SimpleNeuralNetwork {
    /// A list of layers in this network. The last one is the output layer.
    pub layers: Vec<NeuralLayer>,
}

impl SimpleNeuralNetwork {
    /**
     * Constructs a neural network from layer sizes.
     *
     * The first number is actually the input size, rather than a number of
     * neurons proper.
     *
     * A list of activation Options is used. To use the same activation in
     * every layer, see [Self::new_simple_with_activation].
     */
    pub fn new_simple(layer_sizes: &[usize], activations: &[Option<NNActivation>]) -> Self {
        SimpleNeuralNetwork {
            layers: layer_sizes
                .iter()
                .take(layer_sizes.len() - 1)
                .zip(layer_sizes.iter().skip(1))
                .enumerate()
                .map(|item| {
                    let (i, (a, b)) = item;
                    NeuralLayer::new(*a, *b, activations[i])
                })
                .collect(),
        }
    }

    /**
     * Constructs a neural network from layer sizes, reusing the same activation
     * for every layer.
     *
     * The first number is actually the input size, rather than a number of
     * neurons proper.
     */
    pub fn new_simple_with_activation(
        layer_sizes: &[usize],
        activation: Option<NNActivation>,
    ) -> Self {
        Self::new_simple(layer_sizes, vec![activation; layer_sizes.len()].as_slice())
    }

    /// Returns the input size of this network, as determined by its first
    /// layer.
    pub fn input_size(&self) -> Result<usize, String> {
        match self.layers.first() {
            None => Err(
                "There are no layers in this network; input size could not be determined"
                    .to_owned(),
            ),
            Some(layer) => Ok(layer.input_size),
        }
    }

    /// Returns the output size of this network, as determined by its last
    /// layer.
    pub fn output_size(&self) -> Result<usize, String> {
        match self.layers.last() {
            None => Err(
                "There are no layers in this network; output size could not be determined"
                    .to_owned(),
            ),
            Some(layer) => Ok(layer.output_size),
        }
    }

    /// Computes a list of floats and saves the result in an output buffer.
    pub fn compute_values(&self, inputs: &[f32], outputs: &mut [f32]) -> Result<(), String> {
        if cfg!(debug) || cfg!(tests) {
            if self.layers.is_empty() {
                return Err("There are no layers in this network".to_owned());
            }

            if inputs.len() != self.input_size().unwrap() as usize {
                return Err(
                    "The number of input values does not match the input size of this network"
                        .to_owned(),
                );
            }

            if outputs.len() != self.output_size().unwrap() as usize {
                return Err("The size of the destination array does not match the output size of this network".to_owned());
            }
        }

        let mut in_values = inputs.to_vec();

        for layer in &self.layers {
            let mut dest = vec![0.0; layer.output_size as usize];

            layer.compute(&in_values, &mut dest)?;

            in_values = dest;
        }

        outputs.copy_from_slice(&in_values);

        Ok(())
    }
}
