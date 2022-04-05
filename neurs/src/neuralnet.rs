use super::activations::relu;
/**
 * A basic neural network structure.
 *
 * It is quite simplistic, only supporting
 * feed-forward neural networks of dense layers.
 * It also provides a default activation function,
 * the ReLu, although any can be supplied.
 */
use rand::prelude::*;

type NNActivation = fn(f32) -> f32;

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
    pub input_size: u16,

    /// The output size of the layer.
    pub output_size: u16,

    /// The product of the input and output sizes of the layer.
    pub area: u32,
}

impl NeuralLayer {
    /// Create a dense layer with random weights and biases, from an input and output
    /// sizes and an activation function.
    ///
    /// If `activation` is `None`, it will default to [relu].
    pub fn new(input_size: u16, output_size: u16, activation: Option<NNActivation>) -> NeuralLayer {
        let activation = activation.unwrap_or(relu);

        let area: u32 = input_size as u32 * output_size as u32;

        let mut weights: Vec<f32> = vec![0.0; area as usize];
        let mut biases: Vec<f32> = vec![0.0; output_size as usize];

        thread_rng().fill(weights.as_mut_slice());
        thread_rng().fill(biases.as_mut_slice());

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
    pub fn compute(&self, inputs: &[f32], outputs: &mut [f32]) -> Result<(), String> {
        if cfg!(debug) {
            if inputs.len() < self.input_size as usize {
                return Err("Source slice is smaller than the input size of this layer".to_owned());
            }

            if outputs.len() < self.output_size as usize {
                return Err(
                    "Destination slice is smaller than the output size of this layer".to_owned(),
                );
            }
        }

        let input_size = self.input_size;

        for (i, output) in outputs.iter_mut().enumerate() {
            let weight_slice =
                &self.weights[i * self.input_size as usize..(i + 1) * self.input_size as usize];

            *output = (self.activation)(
                self.biases[i]
                    + ((0..input_size as usize)
                        .map(|j| inputs[j] * weight_slice[j])
                        .reduce(|a, b| a + b)
                        .unwrap_or(0.0)),
            )
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
     * every layer, see [new_simple_with_activation].
     */
    pub fn new_simple(layer_sizes: &[u16], activations: &[Option<NNActivation>]) -> Self {
        SimpleNeuralNetwork {
            layers: layer_sizes
                .iter()
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
        layer_sizes: &[u16],
        activation: Option<NNActivation>,
    ) -> Self {
        Self::new_simple(layer_sizes, vec![activation; layer_sizes.len()].as_slice())
    }

    /// Returns the input size of this network, as determined by its first
    /// layer.
    pub fn input_size(&self) -> Result<u16, String> {
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
    pub fn output_size(&self) -> Result<u16, String> {
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
        if cfg!(debug) {
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

        let mut in_values = inputs;
        let mut dest: Vec<f32>;
        let mut dest_values: Vec<f32>;

        for layer in &self.layers {
            dest = vec![0.0; layer.output_size as usize];

            layer.compute(in_values, &mut dest)?;

            dest_values = dest;
            in_values = &dest_values;
        }

        outputs.copy_from_slice(in_values);

        Ok(())
    }
}
