#[cfg(test)]
mod tests {
    use neurs::prelude::*;
    use neurs::train::*;
    use neurs::train::{label, trainer};
    use neurs::{activations, neuralnet};

    fn test_net<MSF, LT>(
        mut classifier: NeuralClassifier,
        test_cases: Vec<Vec<f32>>,
        makes_sense: MSF,
    ) where
        MSF: Fn(&[f32], &[f32]) -> bool,
        LT: TrainingLabel,
    {
        let mut outputs = vec![0.0_f32; LT::num_labels()];

        for inp in &test_cases {
            classifier
                .classifier
                .compute_values(&inp, &mut outputs)
                .unwrap();

            println!(
                "[{}, {}] -> {:?} ([{}, {}])",
                inp[0] as u8,
                inp[1] as u8,
                outputs[1] - outputs[0],
                outputs[0],
                outputs[1]
            );
        }

        println!("Asserting answers make sense...");
        let mut ok_cases = 0;

        for (i, inp) in vec![[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
            .iter()
            .enumerate()
        {
            classifier
                .classifier
                .compute_values(inp, &mut outputs)
                .unwrap();

            let worked = makes_sense(&outputs, inp);

            if worked {
                ok_cases += 1;
                println!("Output in case #{} makes sense.", i + 1);
            } else {
                println!("Output in case #{} does NOT make sense.", i + 1);
            }
        }

        println!("{} out of {} cases make sense.", ok_cases, test_cases.len());
        assert_eq!(ok_cases, test_cases.len());

        println!("Yay!");
    }

    // Test instances

    #[test]
    fn test_jitter_training_xor() {
        let mut net = neuralnet::SimpleNeuralNetwork::new_simple_with_activation(
            &[2, 3, 2],
            Some(activations::fast_sigmoid),
        );

        let mut classifier = NeuralClassifier { classifier: net };

        let mut frame: label::LabeledLearningFrame<bool> = label::LabeledLearningFrame::new(
            vec![
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 1.0],
                vec![0.0, 0.0],
            ],
            vec![true, true, false, false],
            Some(Box::new(|x: f64| x * x)),
        )
        .unwrap();

        let num_cases = frame.num_cases();
        println!("There are {} training cases.", num_cases);

        let strategy = WeightJitterStrat::new(WeightJitterStratOptions {
            apply_bad_jitters: false,
            num_jitters: 100,
            jitter_width: 1.0,
            adaptive_jitter_width: Some(|_jw, mfit, _rfit| 0.01 - mfit * 1.4),
            jitter_width_falloff: 0.0,
            step_factor: 0.6,
            num_steps_per_epoch: num_cases,
        });

        let mut jitter_width = strategy.jitter_width;
        let jitter_width_falloff = strategy.jitter_width_falloff;
        let adaptive_jitter_width = strategy.adaptive_jitter_width.clone();

        let mut trainer = trainer::Trainer::new(&mut classifier, frame, strategy);

        println!("Trainer initialized successfully!");

        println!("Training xor network...");

        for epoch in 1..=250 {
            let ref_fitness = frame
                .avg_reference_fitness(&mut trainer.reference_assembly)
                .unwrap();
            let best_fitness = trainer.epoch().unwrap();

            jitter_width *= 1.0 - jitter_width_falloff;

            if adaptive_jitter_width.is_some() {
                jitter_width = adaptive_jitter_width.as_ref().unwrap()(
                    jitter_width,
                    best_fitness,
                    ref_fitness,
                );
            }

            println!(
                "Epoch {} done! Best fitness {}, jitter width now {}",
                epoch, best_fitness, jitter_width
            );
        }

        println!("Done training! Testing XOR network:");

        test_net(
            trainer,
            vec![
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
                vec![0.0, 0.0],
            ],
            |out: &[f32], inp: &[f32]| {
                (out[1] - out[0]) * (((inp[0] > 0.5) != (inp[1] > 0.5)) as u8 as f32 * 2.0 - 1.0)
                    > 0.5
            },
        );
    }
}
