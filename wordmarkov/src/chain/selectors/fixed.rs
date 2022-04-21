//! Highest and lowest weight selectors.

use crate::prelude::MarkovTraverseDir;

use super::super::token::MarkovToken;
use super::interface::{MarkovSelector, SelectionType};

pub struct StaticBestSelector;

impl MarkovSelector for StaticBestSelector {
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
        SelectionType::Highest
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
        hits: usize,
    ) -> f32 {
        hits as f32
    }

    fn selection_type(&mut self) -> SelectionType {
        SelectionType::Lowest
    }
}
