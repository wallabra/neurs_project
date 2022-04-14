/*!
 * _styliso_ is an experiment which seeks to employ machine learning to
 * distinguish the 'style' between different sets of multiple images.
 * '_styliso_' is short for '_sty_le _iso_lator'.
 *
 * More precisely, it aims to identify potential patterns in each set that
 * distinguish it from the other sets, as well as patterns in common. Part of
 * the goal is to eventually be able to produce a new image to mimick the
 * 'style' of one of said sets.
 *
 * To achieve this, _styliso_ implements an autoencoder neural network
 * implementation, which allows images with corresponding labels to be
 * converted into and back from a 'distilled' representation.
 *
 * That way, by providing a particular label but random image data as input,
 * the autoencoder's weights should hopefully cause the output to retain some
 * of the patterns found in the images, especially in images of that label. It
 * also might or not be possible to mix and match labels by specifying multiple
 * of them in the input.
 */
pub mod autoenc;
pub mod image;
pub mod prelude;

pub fn main() {}
