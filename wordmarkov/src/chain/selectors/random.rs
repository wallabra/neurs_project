//! Random and weighted random selectors.

use crate::prelude::MarkovTraverseDir;

use super::super::body::MarkovToken;
use super::interface::{MarkovSelector, SelectionType};

pub struct WeightedRandomSelector;

impl MarkovSelector for WeightedRandomSelector {
    fn reset(&mut self, _dir: MarkovTraverseDir) {}

    fn weight<'a>(
        &mut self,
        _from: &MarkovToken<'a>,
        _to: &MarkovToken<'a>,
        _punct: &MarkovToken<'a>,
        hits: usize,
    ) -> f32 {
        hits as f32
    }

    fn selection_type(&mut self) -> SelectionType {
        SelectionType::WeightedRandom
    }
}

pub struct NaiveRandomSelector;

impl MarkovSelector for NaiveRandomSelector {
    fn reset(&mut self, _dir: MarkovTraverseDir) {}

    fn weight<'a>(
        &mut self,
        _from: &MarkovToken<'a>,
        _to: &MarkovToken<'a>,
        _punct: &MarkovToken<'a>,
        _hits: usize,
    ) -> f32 {
        1.0
    }

    fn selection_type(&mut self) -> SelectionType {
        SelectionType::WeightedRandom
    }
}
