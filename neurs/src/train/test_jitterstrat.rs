use super::jitterstrat::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::{label, trainer};
    use crate::{activations, neuralnet};

    fn test_net<MSF>(mut trainer: trainer::Trainer, test_cases: Vec<Vec<f32>>, makes_sense: MSF)
    where
        MSF: Fn(&[f32], &[f32]) -> bool,
    {
        let mut outputs: Vec<f32> = vec![0.0; trainer.reference_net.output_size().unwrap()];

        for inp in &test_cases {
            trainer
                .reference_net
                .compute_values(&inp, &mut outputs)
                .unwrap();
            println!(
                "[{}, {}] -> {:?} ([{}, {}]) (fitness {})",
                inp[0] as u8,
                inp[1] as u8,
                outputs[1] - outputs[0],
                outputs[0],
                outputs[1],
                trainer.frame.get_fitness(&inp, &outputs)
            );
        }

        println!("Asserting answers make sense...");
        let mut ok_cases = 0;

        for (i, inp) in vec![[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
            .iter()
            .enumerate()
        {
            trainer
                .reference_net
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
        let net = neuralnet::SimpleNeuralNetwork::new_simple_with_activation(
            &[2, 3, 2],
            Some(activations::fast_sigmoid),
        );

        let mut frame: label::LabeledLearningFrame<usize> = label::LabeledLearningFrame::new(
            vec![
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 1.0],
                vec![0.0, 0.0],
            ],
            vec![1, 1, 0, 0],
            Some(Box::new(|x: f32| x * x)),
            true,
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

        let mut trainer =
            trainer::Trainer::new_from_net(&net, Box::from(frame.clone()), Box::from(strategy));

        println!("Trainer initialized successfully!");

        println!("Training xor network...");

        for epoch in 1..=250 {
            let ref_fitness = frame
                .avg_reference_fitness(&mut trainer.reference_net)
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
