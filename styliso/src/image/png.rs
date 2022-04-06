/*!
 * A module concerned with implementing PNG saving and
 * loading for [image::data::ImageData].
 */
use super::data::ImageData;
use std::io::{Read, Write};

use color_space::{FromRgb, Hsl, Rgb, ToRgb};
use png::BitDepth::*;
use png::{Decoder as PNGDecoder, DecodingError, Encoder as PNGEncoder, EncodingError};

/// A simple error class which encompasses both errors from the `png` crate and basic errors from this crate.
pub enum GenericPngError {
    /// An error coming from `png`'s decoding facilities.
    PngDecodeError(DecodingError),

    /// An error coming from `png`'s encoding facilities.
    PngEncodeError(EncodingError),

    /// An error coming from our own image loading or saving facilities.
    ImageDataError(String),
}

impl ImageData {
    /// Writes ImageData into a PNG file.
    pub fn to_png<W: Write>(&self, output: W) -> Result<(), GenericPngError> {
        let mut writer = PNGEncoder::new(output, self.width as u32, self.height as u32);

        writer.set_depth(Sixteen);

        /* PNGWriter::new(output, PartialInfo {
            width: self.width,
            height: self.height,
            bit_depth: BitDepth.Eight,
        }); */

        let mut rgb_values = vec![0u16; self.area as usize * 3];

        if self.colour.is_some() {
            let colour = self.colour.as_ref().unwrap();

            for i in 0..self.area as usize {
                let rgb = Hsl {
                    h: colour.0[i] as f64,
                    s: colour.1[i] as f64,
                    l: self.brightness[i] as f64,
                }
                .to_rgb();

                rgb_values[i * 3] = (rgb.r as f64 * u16::MAX as f64) as u16;
                rgb_values[i * 3 + 1] = (rgb.g as f64 * u16::MAX as f64) as u16;
                rgb_values[i * 3 + 2] = (rgb.b as f64 * u16::MAX as f64) as u16;
            }
        } else {
            for i in 0..self.area as usize {
                let val = (self.brightness[i] as f64 * u16::MAX as f64) as u16;

                rgb_values[i * 3] = val;
                rgb_values[i * 3 + 1] = val;
                rgb_values[i * 3 + 2] = val;
            }
        }

        let mut rgb_data = vec![0u8; self.area as usize * 6];

        for (idx, val) in rgb_values.iter().enumerate() {
            rgb_data[idx * 2] = (val & 0xFF00 >> 8) as u8;
            rgb_data[idx * 2 + 1] = (val & 0x00FF) as u8;
        }

        let mut datawriter = writer
            .write_header()
            .map_err(GenericPngError::PngEncodeError)?;

        datawriter
            .write_image_data(&rgb_data)
            .map_err(GenericPngError::PngEncodeError)?;
        datawriter
            .finish()
            .map_err(GenericPngError::PngEncodeError)?;

        Ok(())
    }

    //---

    /// Loads ImageData from a PNG file.
    pub fn from_png<R: Read>(input: R) -> Result<Self, GenericPngError> {
        let reader = PNGDecoder::new(input);

        let mut datareader = reader
            .read_info()
            .map_err(GenericPngError::PngDecodeError)?;

        let info = datareader.info();

        let width = info.width;
        let height = info.height;
        let area = width * height;
        let depth = info.bit_depth;

        let mut brightness = vec![0.0f32; area as usize];
        let mut hue = vec![0.0f32; area as usize];
        let mut saturation = vec![0.0f32; area as usize];

        let mut startidx = 0;

        while let Some(row) = datareader
            .next_row()
            .map_err(GenericPngError::PngDecodeError)?
        {
            let rowdata = row.data();

            for x in 0..width as usize {
                let rgb = match depth {
                    Eight => Rgb::new(
                        rowdata[x * 3] as f64 / u8::MAX as f64,
                        rowdata[x * 3 + 1] as f64 / u8::MAX as f64,
                        rowdata[x * 3 + 2] as f64 / u8::MAX as f64,
                    ),

                    Sixteen => Rgb::new(
                        (rowdata[x * 6] as u16 | (rowdata[x * 6 + 1] as u16) << 8) as f64
                            / u16::MAX as f64,
                        (rowdata[x * 6 + 2] as u16 | (rowdata[x * 6 + 3] as u16) << 8) as f64
                            / u16::MAX as f64,
                        (rowdata[x * 6 + 4] as u16 | (rowdata[x * 6 + 5] as u16) << 8) as f64
                            / u16::MAX as f64,
                    ),

                    Four => {
                        if x % 2 == 0 {
                            let bx = x * 3 / 2;

                            Rgb::new(
                                (rowdata[bx] as u8 & 0xF0 >> 4) as f64 / 16_f64,
                                (rowdata[bx] as u8 & 0x0F) as f64 / 16_f64,
                                (rowdata[bx + 1] as u8 & 0xF0 >> 4) as f64 / 16_f64,
                            )
                        } else {
                            let bx = 1 + ((x - 1) * 3 / 2);

                            Rgb::new(
                                (rowdata[bx] as u8 & 0x0F) as f64 / 16_f64,
                                (rowdata[bx] as u8 & 0xF0 >> 4) as f64 / 16_f64,
                                (rowdata[bx + 1] as u8 & 0x0F) as f64 / 16_f64,
                            )
                        }
                    }

                    Two => {
                        return Err(GenericPngError::ImageDataError(
                            "PNGs with bit depth 2 not supported".to_owned(),
                        ));
                    }

                    One => {
                        return Err(GenericPngError::ImageDataError(
                            "PNGs with bit depth 1 not supported".to_owned(),
                        ));
                    }
                };

                let hsl = Hsl::from_rgb(&rgb);

                brightness[startidx + x] = hsl.l as f32;
                hue[startidx + x] = hsl.h as f32;
                saturation[startidx + x] = hsl.s as f32;
            }

            startidx += height as usize;
        }

        Ok(ImageData {
            area,
            brightness,

            width: width as u16,
            height: height as u16,

            colour: Some((hue, saturation)),
        })
    }
}
