/*!
 * The Selector interface, a trait whose implementors can be used as a way of
 * deciding the next state of a Markov chain.
 */

use crate::prelude::MarkovToken;

/**
 * The way in which the weights returned by [MarkovSelector::weight] should be
 * interpreted.
 */
pub enum SelectionType {
    WeightedRandom,
    Highest,
    Lowest,
}
/**
 * An object which can be used as a selector for a Markov chain.
 */
pub trait MarkovSelector {
    /**
     * Reset the state of this MarkovSelector.
     *
     * Must always be called before composing a new sentence.
     */
    fn reset(&mut self);

    /**
     * The weight of a particular link.
     *
     * A numeric value. Can be interpreted as a probability, or as a decision
     * weight, depending on the return value of [selection_type]. If the latter
     * is the case, the highest (or lowest!) weight is always the final
     * selection.
     *
     * Stateful.
     */
    fn weight<'a>(
        &mut self,
        from: &MarkovToken<'a>,
        to: &MarkovToken<'a>,
        punct: &MarkovToken<'a>,
        occurrences: usize,
    ) -> f32;

    /**
     * Returns the [SelectionType] of this Selector; this will decide how the
     * weight returned by [weight] should be interpreted.
     */
    fn selection_type(&mut self) -> SelectionType;
}
