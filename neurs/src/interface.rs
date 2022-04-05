/**
 * A generic interface for autoencoder behaviour.
 *
 * This is actually implemented by the [neuralnet] module.
 */

/**
 * A basic interface for any item that can be autoencoded.
 */
pub trait Item {
    /**
     * Encode this item into a vector of floats.
     *
     * Used by _styliso_'s autoencoder neural network logic.
     */
    fn encode(&self) -> Result<Vec<f32>, &str>;

    /**
     * Decode into an item of this type, from a vector of floats.
     *
     * Used by _styliso_'s autoencoder neural network logic.
     */
    fn decode_from(&mut self, input: &[f32]) -> Result<(), String>;
}

/**
 * The basic interface for an object that can behave as an Autoencoder.
 *
 * Usually, you're looking at a neural network from the [neuralnet] module of
 * _styliso_.
 */
pub trait Autoencoder<T: Item> {
    /// "Implodes" an item into a distilled representation of f32.
    fn implode(&self, item: &T) -> Vec<f32>;

    /// "Explodes" a distilled representation into an item.
    fn explode(&self, imploded: &[f32]) -> T;
}
