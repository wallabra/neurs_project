#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;
    use neurs::prelude::*;

    #[test]
    fn layer_output_makes_sense() {
        let mut some_layer = NeuralLayer::new(4, 1, Some(activations::relu));

        let mut out_test: [f32; 1] = [0.0f32];

        let mut case = |weights: [f32; 4], biases: [f32; 1], inputs: [f32; 4], result| {
            some_layer.weights.copy_from_slice(&weights);
            some_layer.biases.copy_from_slice(&biases);

            let res = some_layer.compute(&inputs, &mut out_test);

            assert!(res.is_ok());

            assert_float_eq!(out_test[0], result, abs <= 2.0 * f32::EPSILON);
        };

        case([-2.0, 3.0, 0.0, 1.0], [0.0], [15.0, 2.0, 3.0, 4.0], 0.0);
        case([-2.0, 3.0, 0.0, 1.0], [4.0], [1.0, 2.0, 3.0, 4.0], 12.0);
        case([2.0, -3.0, 0.0, 2.0], [-1.0], [1.0, 2.0, 3.0, 4.0], 3.0);
        case([2.0, -3.0, 0.0, 2.0], [-16.0], [1.0, 1.0, 3.0, 6.0], 0.0);
    }
}
