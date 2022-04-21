/*!
 * Word Vectorizer.
 *
 * The neural network which converts variable-length words into a fixed-length
 * vector representation.
 */

use neurs::prelude::*;

/**
 * An assembly of two neural networks which can boil a word down to a fixed length
 * vector.
 */
pub struct WordVectorizer {
    encoder: SimpleNeuralNetwork,
    decoder: SimpleNeuralNetwork,

    conv_order: usize,
    alphabet: String,
    alphabet_size: usize,
    out_vec_size: usize,
}

const DEFAULT_ALPHABET: &str =
    ",.!?;:_-=+()[]{}/\\ 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

impl Default for WordVectorizer {
    fn default() -> WordVectorizer {
        WordVectorizer::new(5, 10, None, None)
    }
}

impl WordVectorizer {
    pub fn new(
        conv_order: usize,
        out_vec_size: usize,
        alphabet: Option<String>,
        activation: Option<NNActivation>,
    ) -> WordVectorizer {
        let alphabet = alphabet.unwrap_or_else(|| DEFAULT_ALPHABET.to_string());
        let alphabet_size = alphabet.len();

        WordVectorizer {
            conv_order,
            alphabet,
            alphabet_size,
            out_vec_size,

            encoder: SimpleNeuralNetwork::new_simple_with_activation(
                &[
                    2 + alphabet_size * conv_order + out_vec_size,
                    2 * alphabet_size * conv_order,
                    3 * out_vec_size,
                    out_vec_size,
                ],
                activation.or(Some(activations::fast_sigmoid)),
            ),

            decoder: SimpleNeuralNetwork::new_simple_with_activation(
                &[
                    2 + out_vec_size,
                    3 * out_vec_size,
                    2 * (out_vec_size + alphabet_size),
                    alphabet_size + out_vec_size,
                ],
                activation.or(Some(activations::fast_sigmoid)),
            ),
        }
    }

    fn set_closeness(&self, inputs: &mut [f32], curr: usize, len: usize) {
        let far: f32 = if len > 1 {
            curr as f32 / len as f32
        } else {
            0.0
        };

        let near: f32 = 1.0 - far;

        inputs[0] = near;
        inputs[1] = far;
    }

    fn set_char_one_hot(&self, inputs: &mut [f32], ch: char) {
        match self
            .alphabet
            .char_indices()
            .position(|(_pos, char)| char == ch)
        {
            Some(pos) => inputs[pos] = 1.0,

            None => {}
        };
    }

    fn convolve_one(
        &self,
        inputs: &mut [f32],
        curr_out: &mut [f32],
        word: &str,
        curr: usize,
    ) -> Result<(), String> {
        let len = word.len();
        let last_len = inputs.len() - self.alphabet_size;

        self.set_closeness(&mut inputs[..2], curr, len);
        self.encoder.compute_values(inputs, curr_out)?;
        inputs[2 + self.conv_order * self.alphabet_size..].copy_from_slice(curr_out);

        inputs[2..].rotate_left(self.alphabet_size);
        inputs[2 + last_len..].fill(0.0_f32);

        Ok(())
    }

    pub fn encode(&self, word: &str) -> Result<Vec<f32>, String> {
        let mut inputs =
            vec![0.0_f32; 2 + self.conv_order * self.alphabet_size + self.out_vec_size];

        let mut curr_out = vec![0.0_f32; self.out_vec_size];

        for init_ch in word[..self.conv_order].chars() {
            self.set_char_one_hot(&mut inputs[2..], init_ch);
        }

        self.convolve_one(&mut inputs, &mut curr_out, word, 0)?;

        let new_chars = word[self.conv_order..].chars().enumerate();
        let last_len = inputs.len() - self.alphabet_size;

        for (i, char) in new_chars {
            self.set_char_one_hot(&mut inputs[2 + last_len..], char);

            assert!(i < word.len() - self.conv_order);
            self.convolve_one(&mut inputs, &mut curr_out, word, i + 1)?;
        }

        Ok(curr_out)
    }

    pub fn decode_one_char(
        &self,
        inputs: &mut [f32],
        curr_vec: &mut [f32],
        curr_out: &mut [f32],
        idx: usize,
        len: usize,
    ) -> Result<char, String> {
        self.set_closeness(inputs, idx, len);
        inputs[2..].copy_from_slice(curr_vec);

        self.decoder.compute_values(inputs, curr_out)?;

        curr_vec.copy_from_slice(&curr_out[self.alphabet_size..]);

        Ok(self
            .alphabet
            .chars()
            .zip(curr_out[..self.alphabet_size].iter())
            .reduce(|(lch, lval), (ch, val)| if val > lval { (ch, val) } else { (lch, lval) })
            .unwrap() // Assume alphabet is never an empty string
            .0)
    }

    pub fn decode(&self, vec: &[f32], len: usize) -> Result<String, String> {
        assert_eq!(vec.len(), self.out_vec_size);

        let mut res: Vec<char> = vec![' '; len];
        let mut inputs: Vec<f32> = vec![0.0_f32; 2 + self.out_vec_size];
        let mut outputs: Vec<f32> = vec![0.0_f32; self.alphabet_size + self.out_vec_size];
        let mut curr_in: Vec<f32> = vec.to_vec();

        for (idx, rval) in res.iter_mut().enumerate() {
            *rval = self.decode_one_char(&mut inputs, &mut curr_in, &mut outputs, idx, len)?;
        }

        Ok(res.iter().collect())
    }
}
