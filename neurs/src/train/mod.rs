/**
 * Neural network training facilities.
 *
 * Provides an interface for training strategies and rules,
 * as well as a simple implementation,
 */
pub mod interface;
pub mod jitterstrat;
pub mod label;
pub mod trainer;

pub use interface::*;
pub use trainer::*;
