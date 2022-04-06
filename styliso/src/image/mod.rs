/*!
 * Organizes image related code, especially
 * concerning PNG files and the implementation
 * of autoencoder traits for image data.
 */
pub mod data;
pub mod labeled;
pub mod png;

pub use data::*;
pub use labeled::*;
