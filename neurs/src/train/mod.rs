/*!
 * Neural network training facilities.
 *
 * Provides an interface for training strategies and rules,
 * as well as a simple implementation,
 */
pub mod interface;
pub mod jitterstrat;
pub mod label;
pub mod prelude;
pub mod trainer;

pub mod prelude
    pub use super::interface::*;
    pub use super::jitterstrat::*;
    pub use super::label::*;
    pub use super::trainer::*;
}
