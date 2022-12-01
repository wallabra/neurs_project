/*!
 * Actual Markov chain container.
 */

use super::selectors::interface::MarkovSelector;
use super::selectors::interface::SelectionType;
use super::token::*;
use crate::sentence::lex::{Lexer, Token as LexedToken};
use rand::{distributions::Uniform, prelude::*};
use std::collections::HashMap;
use std::collections::LinkedList;
use std::rc::Rc;

/// The direction in which to traverse the Markov chain.
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum MarkovTraverseDir {
    Forward,
    Reverse,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MarkovSeed<'a> {
    Word(&'a str),
    Id(usize),
    Random,
}

/// An edge linking two words in the Markov chain.
pub struct Edge {
    /// The word this edge comes from.
    pub src_idx: usize,

    /// The word this edge leads into.
    pub dst_idx: usize,

    /// How many times this edge has been found.
    pub hits: usize,

    /// The punctuation in this edge.
    pub pct_idx: usize,
}

impl Edge {
    /// Get the word from which this Edge sprouts.
    pub fn get_source<'a>(&self, chain: &'a MarkovChain) -> MarkovToken<'a> {
        chain.get_textlet(self.src_idx).unwrap()
    }

    /// Get the word into which this Edge leads.
    pub fn get_dest<'a>(&self, chain: &'a MarkovChain) -> MarkovToken<'a> {
        chain.get_textlet(self.dst_idx).unwrap()
    }

    /// Get the punctuation between the words this Edge connects.
    pub fn get_punct<'a>(&self, chain: &'a MarkovChain) -> MarkovToken<'a> {
        chain.get_textlet(self.pct_idx).unwrap()
    }
}

/**
 * A graph that links tokens together.
 */
pub struct MarkovChain {
    textlet_bag: Vec<MarkovTokenOwned>,
    textlet_indices: HashMap<Rc<str>, usize>,

    edge_list: Vec<Edge>,
    edges: HashMap<usize, Vec<usize>>,
    reverse_edges: HashMap<usize, Vec<usize>>,

    seedbag: Vec<usize>,
}

impl Default for MarkovChain {
    fn default() -> Self {
        Self::new()
    }
}

impl MarkovChain {
    /**
     * Makes a new empty [MarkovChain].
     */
    pub fn new() -> MarkovChain {
        MarkovChain {
            textlet_bag: vec![MarkovTokenOwned::Begin, MarkovTokenOwned::End],
            textlet_indices: HashMap::new(),

            edge_list: Vec::new(),
            edges: HashMap::new(),
            reverse_edges: HashMap::new(),

            seedbag: Vec::new(),
        }
    }

    /**
     * Gets the index of a textlet in this chain; if the textlet is not found,
     * makes a new one and returns that instead.
     */
    pub fn ensure_textlet_index(&mut self, word: &str) -> usize {
        match self.textlet_indices.get(word) {
            Some(a) => *a,
            None => {
                let i = self.textlet_bag.len();
                let rcword: Rc<str> = Rc::from(word);

                self.textlet_bag
                    .push(MarkovTokenOwned::Textlet(rcword.clone()));

                self.textlet_indices.insert(rcword, i);

                i
            }
        }
    }

    /**
     * Get a textlet index from a [crate::sentence::lex::Token].
     *
     * If one does not exist, make one and return that instead.
     */
    pub fn ensure_textlet_from_token(&mut self, token: LexedToken) -> usize {
        match token {
            LexedToken::Begin => 0,
            LexedToken::End => 1,
            LexedToken::Punct(word) => self.ensure_textlet_index(word),
            LexedToken::Word(word) => self.ensure_textlet_index(word),
        }
    }

    /**
     * Tries to get the index of a textlet in this chain.
     *
     * If the textlet is not registered, returns None.
     */
    pub fn try_get_textlet_index(&self, word: &str) -> Option<usize> {
        self.textlet_indices.get(word).copied()
    }

    /**
     * Gets the [MarkovToken] of a textlet by its index.
     */
    pub fn get_textlet(&self, index: usize) -> Option<MarkovToken<'_>> {
        self.textlet_bag.get(index).map(MarkovToken::from)
    }

    fn push_new_edge(
        &mut self,
        from: usize,
        to: usize,
        punct: usize,
        hits: Option<usize>,
    ) -> usize {
        let edge = Edge {
            src_idx: from,
            dst_idx: to,
            hits: hits.unwrap_or(1),
            pct_idx: punct,
        };

        let idx = self.edge_list.len();
        self.edge_list.push(edge);

        idx
    }

    fn add_reverse_edge(&mut self, edge_idx: usize) {
        let edge = &self.edge_list[edge_idx];

        match self.reverse_edges.get_mut(&edge.dst_idx) {
            None => {
                let rev_vec = vec![edge_idx];

                self.reverse_edges.insert(edge.dst_idx, rev_vec);
            }

            Some(rev_vec) => {
                for oedge in rev_vec.iter() {
                    let oedge = self.edge_list.get(*oedge).unwrap();

                    if edge.src_idx == oedge.src_idx && edge.pct_idx == oedge.pct_idx {
                        return;
                    }
                }

                rev_vec.push(edge_idx);
            }
        }
    }

    /**
     * Register a new edge between two word tokens in this chain.
     *
     * `from` and `to` must be existing textlet indices. Same with
     * `punct` – it must be an existing index, and not a space.
     */
    fn register_edge(&mut self, from: usize, to: usize, punct: usize) {
        if !self.seedbag.contains(&from) {
            self.seedbag.push(from);
        }

        if let Some(edgevec) = self.edges.get_mut(&from) {
            for edge in edgevec.iter() {
                let edge: &mut Edge = self.edge_list.get_mut(*edge).unwrap();

                if edge.dst_idx == to && edge.pct_idx == punct {
                    edge.hits += 1;
                    return;
                }
            }
        }

        let idx = self.push_new_edge(from, to, punct, None);
        self.edges.insert(from, vec![idx]);

        if let Some(edgevec) = self.edges.get_mut(&from) {
            edgevec.push(idx);
        } else {
            self.edges.insert(from, vec![idx]);
        }

        self.add_reverse_edge(idx);
    }

    fn get_seed<T: Rng>(&self, seed: MarkovSeed, rng: &mut T) -> Result<usize, String> {
        use MarkovSeed::*;

        match seed {
            Word(seed) => {
                let from = self.try_get_textlet_index(seed);

                if from.is_none() {
                    return Err(format!(
                        "Seed word {:?} not found in this Markov chain!",
                        seed
                    ));
                }

                Ok(from.unwrap())
            }

            Id(seed) => Ok(seed),

            Random => {
                let from: usize = Uniform::new(0, self.seedbag.len()).sample(rng);
                Ok(self.seedbag[from])
            }
        }
    }

    fn _weighted_select<R>(
        &self,
        sel_type: SelectionType,
        edges: &[usize],
        weights: &[f32],
        rng: &mut R,
    ) -> &Edge
    where
        R: Rng,
    {
        match sel_type {
            SelectionType::Lowest => {
                edges
                    .iter()
                    .map(|e| &self.edge_list[*e])
                    .zip(weights.iter())
                    .reduce(|ewc, ewn| if ewc.1 < ewn.1 { ewc } else { ewn })
                    .unwrap()
                    .0
            }

            SelectionType::Highest => {
                edges
                    .iter()
                    .map(|e| &self.edge_list[*e])
                    .zip(weights.iter())
                    .reduce(|ewc, ewn| if ewc.1 > ewn.1 { ewc } else { ewn })
                    .unwrap()
                    .0
            }

            SelectionType::WeightedRandom => {
                let total: f32 = weights.iter().sum();
                let pick = Uniform::new(0.0_f32, total).sample(rng);

                let mut curr = 0.0;
                let mut res = None;

                for (edge, weight) in edges
                    .iter()
                    .map(|e| &self.edge_list[*e])
                    .zip(weights.iter())
                {
                    curr += weight;

                    if curr >= pick {
                        res = Some(edge);
                        break;
                    }
                }

                res.unwrap()
            }
        }
    }

    /**
     * Selects the word following the current one (`from`) based om the
     * criteria of a [MarkovSelector] (`selector`).
     *
     * Returns a tuple (`dest`, `inbetween`, `dest_idx`, `inbetween_idx`).
     * The first two items can be converted into strings because MarkovToken
     * has Into<&str>. The last two items are the corresponding internal
     * indices, which can be reused in functions which take `usize`.
     *
     * `inbetween` is all of the whitespace and punctuation lying between
     * `from` and `dest`. Simply concatenate `from` with `inbetween.into()`
     * with `dest.into()`, in that order.
     */
    pub fn select_next_word(
        &self,
        seed: MarkovSeed,
        selector: &mut dyn MarkovSelector,
        direction: MarkovTraverseDir,
    ) -> Result<(MarkovToken<'_>, MarkovToken<'_>, usize, usize), String> {
        use MarkovTraverseDir::*;

        let mut rng = thread_rng();

        let from: usize = self.get_seed(seed, &mut rng)?;

        let edges = match direction {
            MarkovTraverseDir::Forward => self.edges.get(&from),
            MarkovTraverseDir::Reverse => self.reverse_edges.get(&from),
        };

        if edges.is_none() {
            return Err(format!(
                "Seed textlet {:?} is not connected to anything in this Markov chain!",
                self.get_textlet(from)
            ));
        }

        let edges = edges.unwrap();

        if edges.is_empty() {
            return Err(format!("Seed textlet {:?} is not connected to anything in this Markov chain, but in a weird way!", self.get_textlet(from)));
        }

        let mut weights: Vec<f32> = vec![0.0; edges.len()];

        selector.reset(direction);

        for (edge, weight) in edges
            .iter()
            .map(|e| &self.edge_list[*e])
            .zip(weights.iter_mut())
        {
            *weight = selector.weight(
                &edge.get_source(self),
                &edge.get_dest(self),
                &edge.get_punct(self),
                edge.hits,
            );
        }

        let sel_type = selector.selection_type();

        let best_edge: &Edge = self._weighted_select(sel_type, edges, &weights, &mut rng);

        match direction {
            Forward => Ok((
                best_edge.get_dest(self),
                best_edge.get_punct(self),
                best_edge.dst_idx,
                best_edge.pct_idx,
            )),

            Reverse => Ok((
                best_edge.get_source(self),
                best_edge.get_punct(self),
                best_edge.src_idx,
                best_edge.pct_idx,
            )),
        }
    }

    /**
     * The number of textlets in this chain.
     *
     * Includes unique instances of whitespace or punctuation, as well as the
     * internal tokens [MarkovTokenOwned::Begin] and [MarkovTokenOwned::End].
     */
    pub fn num_textlets(&self) -> usize {
        self.textlet_bag.len()
    }

    /**
     * The number of [Edge]s registered in this chain.
     *
     * Includes edges connected to the internal tokens
     * [MarkovTokenOwned::Begin] and [MarkovTokenOwned::End].
     */
    pub fn num_edges(&self) -> usize {
        self.edge_list.len()
    }

    /**
     * Parse a sentence, registering textlets and edges
     * for it.
     */
    pub fn parse_sentence(&mut self, sentence: &str) {
        let mut lexer = Lexer::new(sentence);
        let mut curr_token = lexer.next();

        let mut to_register: Vec<(LexedToken, LexedToken, LexedToken)> = vec![];

        if sentence.is_empty() {
            return;
        }

        loop {
            if curr_token.is_none() {
                panic!("Found a none token prematurely!");
            }

            let token = curr_token.unwrap();

            let punct = lexer.next();
            let next_token = lexer.next();

            if punct.is_none() || next_token.is_none() {
                return;
            }

            let punct = punct.unwrap();
            let next_token = next_token.unwrap();

            to_register.push((token, punct, next_token.clone()));

            if next_token == LexedToken::End {
                break;
            }

            curr_token = Some(next_token);
        }

        for (src, pct, dst) in to_register {
            let src = self.ensure_textlet_from_token(src);
            let pct = self.ensure_textlet_from_token(pct);
            let dst = self.ensure_textlet_from_token(dst);

            self.register_edge(src, dst, pct);
        }
    }

    /// Get the textlet identifier for [MarkovTokenOwned::Begin].
    pub fn begin(&self) -> usize {
        self.textlet_bag
            .iter()
            .position(|a| a == &MarkovTokenOwned::Begin)
            .unwrap()
    }

    /// Get the textlet identifier for [MarkovTokenOwned::End].
    pub fn end(&self) -> usize {
        self.textlet_bag
            .iter()
            .position(|a| a == &MarkovTokenOwned::End)
            .unwrap()
    }

    /// Get the size of the textlet_bag of this Markov chain.
    pub fn len(&self) -> usize {
        self.textlet_bag.len()
    }

    /// Returns whether the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.edge_list.is_empty()
    }

    /**
     * Composes a sentence by traversing this chain forward and backward from a
     * given 'seed word'.
     */
    pub fn compose_sentence<'a>(
        &'a self,
        seed: MarkovSeed,
        selector: &mut dyn MarkovSelector,
        max_len: Option<usize>,
    ) -> Result<TokenList<'a>, String> {
        use MarkovSeed::Id;
        use MarkovToken::*;
        use MarkovTraverseDir::*;

        let mut rng = thread_rng();

        if self.is_empty() {
            return Err("Cannot compose a sentence from an empty chain".into());
        }

        let seed = self.get_seed(seed, &mut rng)?;

        let mut sentence: LinkedList<MarkovToken<'a>> =
            LinkedList::from([self.get_textlet(seed).unwrap()]);

        let mut len = self.get_textlet(seed).unwrap().len();

        let mut curr_backward = seed;
        let mut curr_forward = seed;

        let capped = max_len.is_some();
        let max_half_len: Option<usize> = max_len.map(|x| x / 2);

        while curr_backward != self.begin() {
            let (prev, punct, prvidx, _) =
                self.select_next_word(Id(curr_backward), selector, Reverse)?;

            let new_len = len + punct.len() + prev.len();

            if capped && new_len > max_half_len.unwrap() {
                break;
            }

            len = new_len;

            sentence.push_front(punct);

            if prev == Begin {
                break;
            }

            sentence.push_front(prev);

            curr_backward = prvidx;
        }

        while curr_forward != self.begin() {
            let (next, punct, nxtidx, _) =
                self.select_next_word(Id(curr_forward), selector, Forward)?;

            let new_len = len + punct.len() + next.len();

            if capped && new_len > max_len.unwrap() {
                break;
            }

            len = new_len;

            sentence.push_back(punct);

            if next == End {
                break;
            }

            sentence.push_back(next);

            curr_forward = nxtidx;
        }

        Ok(TokenList(sentence))
    }
}
