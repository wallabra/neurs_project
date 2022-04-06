/*!
 * Labeling for images to be autoencoded, which is
 * very particular to this project.
 */
use super::data::ImageData;
use neurs::train::label::TrainingLabel;

/// An image which has been given a label.
pub struct LabeledImage<LabelType: TrainingLabel> {
    label: LabelType,
    img: ImageData,
}

impl<LabelType: TrainingLabel> neurs::Item for LabeledImage<LabelType> {
    /// Vectorizes an image, along with label information, for autoencoding.
    fn encode(&self) -> Result<Vec<f32>, &str> {
        let mut one_hot: Vec<f32> = vec![0.0; LabelType::num_labels() as usize];
        one_hot[self.label.index() as usize] = 1.0;

        let mut res = self.img.encode()?;
        res.append(&mut one_hot);

        Ok(res)
    }

    /// De-vectorizes an image, along with label information, from autoencoder output,
    /// into the values of a LabeledImage.
    fn decode_from(&mut self, input: &[f32]) -> Result<(), String> {
        let img_data_len = input.len() - LabelType::num_labels() as usize;

        let img_data = &input[..input.len() - img_data_len];
        let label_data = &input[img_data_len..];

        assert!(!label_data.is_empty());

        let label_idx = label_data
            .iter()
            .enumerate()
            .reduce(|best, curr| if curr.1 > best.1 { best } else { curr })
            .unwrap()
            .0;

        self.label = LabelType::from_index(label_idx as u16);
        self.img.decode_from(img_data)?;

        Ok(())
    }
}
