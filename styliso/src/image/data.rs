/*!
 * The internal image data holder.
 */
use neurs::interface as autoencoder;

/**
 * Image data, internally represented as separate Vecs
 * of H, S and L values.
 */
pub struct ImageData {
    /// The brightnesses of each pixel.
    /// This is equivalent to the L (luminance) in the HSL
    /// model.
    pub brightness: Vec<f32>,

    /// Optionally, the hue and saturation of each pixel.
    /// If set to None, the image is interpreted as grayscale;
    /// that is, the H and S in HSL are assumed to always be 0.
    pub colour: Option<(Vec<f32>, Vec<f32>)>,

    /// The width of this image.
    pub width: u16,

    /// The height of this image.
    pub height: u16,

    /// The area of this image; that is, the product of its width and height.
    pub area: u32,
}

impl autoencoder::Item for ImageData {
    /// Encodes an image into autoencoder data.
    fn encode(&self) -> Result<Vec<f32>, &str> {
        let area = self.area;

        let res_size: u32 = if self.colour.is_some() {
            area * 3
        } else {
            area
        };

        let mut res: Vec<f32> = vec![0.0f32; res_size as usize];

        // encode brightnesses
        res.copy_from_slice(&self.brightness);

        // encode colours
        if self.colour.is_some() {
            let colour = self.colour.as_ref().unwrap();

            for i in 0..area as usize {
                res[area as usize + i] = colour.0[i]
            }

            for i in 0..area as usize {
                res[2 * area as usize + i] = colour.1[i]
            }
        }

        Ok(res)
    }

    /// Decodes an image from autoencoder output into the values of an ImageData.
    fn decode_from(&mut self, input: &[f32]) -> Result<(), String> {
        let area = self.area;

        let has_colour: bool = if input.len() == area as usize * 3 {
            true
        } else if input.len() == area as usize {
            false
        } else {
            return Err("Incompatible size; array length must be equal to self.area for brightness values, or twice it for brightness and 'colour'".to_owned());
        };

        self.brightness.copy_from_slice(input);

        if has_colour {
            let colour = self.colour.as_mut().unwrap();

            for i in 0..area as usize {
                colour.0[i] = input[area as usize + i]
            }

            for i in 0..area as usize {
                colour.1[i] = input[2 * area as usize + i]
            }
        }

        Ok(())
    }
}
