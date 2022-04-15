/*!
 * Actual Markov chain container.
 */

use super::selectors::interface::MarkovSelector;
use super::selectors::interface::SelectionType;
use crate::sentence::lex::{Lexer, Token as LexedToken};
use rand::{distributions::Uniform, prelude::*};
use std::collections::HashMap;

/// A Markov token.
#[derive(Debug)]
pub enum MarkovToken<'a> {
    Begin,
    End,
    Textlet(&'a str),
}

impl<'a> From<&LexedToken<'a>> for MarkovToken<'a> {
    fn from(lexed: &LexedToken<'a>) -> Self {
        match lexed {
            LexedToken::Begin => MarkovToken::Begin,
            LexedToken::End => MarkovToken::End,

            LexedToken::Punct(w) => MarkovToken::Textlet(w),
            LexedToken::Word(w) => MarkovToken::Textlet(w),
        }
    }
}

/// A Markov token, but owned. Only used from MarkovChain.
#[derive(Debug)]
enum MarkovTokenOwned {
    Begin,
    End,
    Textlet(String),
}

impl<'a> From<&'a MarkovTokenOwned> for MarkovToken<'a> {
    fn from(owned: &'a MarkovTokenOwned) -> Self {
        match owned {
            MarkovTokenOwned::Textlet(s) => MarkovToken::Textlet(s),
            MarkovTokenOwned::Begin => MarkovToken::Begin,
            MarkovTokenOwned::End => MarkovToken::End,
        }
    }
}

impl<'a> From<&'a MarkovTokenOwned> for &'a str {
    fn from(token: &'a MarkovTokenOwned) -> Self {
        match token {
            MarkovTokenOwned::Textlet(s) => s,
            MarkovTokenOwned::Begin => "",
            MarkovTokenOwned::End => "",
        }
    }
}

impl<'a> From<MarkovToken<'a>> for &'a str {
    fn from(token: MarkovToken<'a>) -> Self {
        match token {
            MarkovToken::Textlet(s) => s,
            MarkovToken::Begin => "",
            MarkovToken::End => "",
        }
    }
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
pub struct MarkovChain<'a> {
    textlet_bag: Vec<MarkovTokenOwned>,
    textlet_indices: HashMap<&'a str, usize>,
    words: Vec<usize>,

    edges: HashMap<usize, Vec<Edge>>,
    reverse_edges: HashMap<usize, Vec<&'a Edge>>,
}

impl<'a> Default for MarkovChain<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> MarkovChain<'a> {
    /**
     * Makes a new empty [MarkovChain].
     */
    pub fn new() -> MarkovChain<'a> {
        MarkovChain {
            textlet_bag: vec![MarkovTokenOwned::Begin, MarkovTokenOwned::End],

            textlet_indices: HashMap::new(),

            words: Vec::new(),
            edges: HashMap::new(),
            reverse_edges: HashMap::new(),
        }
    }

    /**
     * Gets the index of a textlet in this chain; if the textlet is not found,
     * makes a new one and returns that instead.
     */
    pub fn ensure_textlet_index<'b>(&'b mut self, word: &str) -> usize
    where
        'b: 'a,
    {
        match self.textlet_indices.get(word) {
            Some(a) => *a,
            None => {
                let i = self.textlet_bag.len();

                self.textlet_bag
                    .push(MarkovTokenOwned::Textlet(word.to_string()));

                let ownedtoken = self.textlet_bag.last().unwrap();

                if let MarkovTokenOwned::Textlet(s) = ownedtoken {
                    self.textlet_indices.insert(s, i);

                    return i;
                }

                unreachable!()
            }
        }
    }

    /**
     * Get a textlet index from a [crate::sentence::lex::Token].
     *
     * If one does not exist, make one and return that instead.
     */
    pub fn ensure_textlet_from_token<'b>(&'b mut self, token: LexedToken<'b>) -> usize
    where
        'b: 'a,
    {
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
    pub fn try_get_textlet_index(&'a self, word: &str) -> Option<usize> {
        self.textlet_indices.get(word).copied()
    }

    /**
     * Gets the [MarkovToken] of a textlet by its index.
     */
    pub fn get_textlet(&'a self, index: usize) -> Option<MarkovToken<'a>> {
        self.textlet_bag.get(index).map(|a| MarkovToken::from(a))
    }

    /**
     * Register a new edge between two word tokens in this chain.
     *
     * `from` and `to` must be existing textlet indices. Same with
     * `punct` – it must be an existing index, and not a space.
     *
     * For both `from` and `to`, if the index is not found in the
     * `self.words` list, it will be added to it.
     */
    pub fn register_edge(&'a mut self, from: usize, to: usize, punct: usize) {
        for item in [from, to] {
            if !self.words.contains(&item) {
                self.words.push(item);
            }
        }

        if let Some(edgevec) = self.edges.get_mut(&from) {
            for edge in edgevec.iter_mut() {
                if edge.dst_idx == to && edge.pct_idx == punct {
                    edge.hits += 1;
                    return;
                }
            }

            edgevec.push(Edge {
                src_idx: from,
                dst_idx: to,
                hits: 1,
                pct_idx: punct,
            });

            let edge = self.edges.get(&from).unwrap().last().unwrap();

            match self.reverse_edges.get_mut(&edge.dst_idx) {
                None => {
                    let rev_vec = vec![edge];

                    self.reverse_edges.insert(edge.dst_idx, rev_vec);
                }

                Some(rev_vec) => {
                    for oedge in rev_vec.iter() {
                        if edge.src_idx == oedge.src_idx && edge.pct_idx == oedge.pct_idx {
                            return;
                        }
                    }

                    rev_vec.push(edge);
                }
            }
        } else {
            let edge = Edge {
                src_idx: from,
                dst_idx: to,
                hits: 1,
                pct_idx: punct,
            };

            self.edges.insert(from, vec![edge]);

            let edge = self.edges.get(&from).unwrap().last().unwrap();

            match self.reverse_edges.get_mut(&edge.dst_idx) {
                None => {
                    let rev_vec = vec![edge];

                    self.reverse_edges.insert(edge.dst_idx, rev_vec);
                }

                Some(rev_vec) => {
                    for oedge in rev_vec.iter() {
                        if edge.src_idx == oedge.src_idx && edge.pct_idx == oedge.pct_idx {
                            return;
                        }
                    }

                    rev_vec.push(edge);
                }
            }
        }
    }

    /**
     * Selects the word following the current one (`from`) based om the
     * criteria of a [MarkovSelector] (`selector`).
     *
     * Returns a tuple (`dest`, `inbetween`) - the second item is a mix of
     * whitespace and punctuation lying between `from` and `dest`.
     *
     * Simply concatenate `from` with `inbetween` with `dest.into()`, in
     * that order.
     */
    pub fn select_next_word(
        &'a self,
        seed: Option<&str>,
        selector: &mut dyn MarkovSelector,
    ) -> Result<(MarkovToken<'a>, &str), String> {
        let mut rng = thread_rng();

        let from: usize = if let Some(seed) = seed {
            let from = self.try_get_textlet_index(seed);

            if from.is_none() {
                return Err(format!(
                    "Seed word {:?} not found in this Markov chain!",
                    seed
                ));
            }

            from.unwrap()
        } else {
            let from: usize = Uniform::new(0, self.words.len()).sample(&mut rng);
            self.words[from]
        };

        let edges = self.edges.get(&from);

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

        let mut weights: Vec<f32> = Vec::with_capacity(edges.len());

        selector.reset();

        for (edge, weight) in edges.iter().zip(weights.iter_mut()) {
            *weight = selector.weight(
                &edge.get_source(self),
                &edge.get_dest(self),
                &edge.get_punct(self),
                edge.hits,
            );
        }

        let sel_type = selector.selection_type();

        let best_edge: &'a Edge = match sel_type {
            SelectionType::Lowest => {
                edges
                    .iter()
                    .zip(weights.iter())
                    .reduce(|ewc, ewn| if ewc.1 < ewn.1 { ewc } else { ewn })
                    .unwrap()
                    .0
            }

            SelectionType::Highest => {
                edges
                    .iter()
                    .zip(weights.iter())
                    .reduce(|ewc, ewn| if ewc.1 > ewn.1 { ewc } else { ewn })
                    .unwrap()
                    .0
            }

            SelectionType::WeightedRandom => {
                let total: f32 = weights.iter().sum();
                let pick = Uniform::new(0.0_f32, total).sample(&mut rng);

                let mut curr = 0.0;
                let mut res = None;

                for (edge, weight) in edges.iter().zip(weights.iter()) {
                    curr += weight;

                    if curr >= pick {
                        res = Some(edge);
                    }
                }

                res.unwrap()
            }
        };

        Ok((
            best_edge.get_dest(self),
            match best_edge.get_punct(self) {
                MarkovToken::Textlet(a) => a,
                _ => unreachable!(),
            },
        ))
    }

    /**
     * Parse a sentence, registering textlets and edges
     * for it.
     */
    pub fn parse_sentence(&'a mut self, sentence: &'a str) {
        let mut lexer = Lexer::new(sentence);
        let mut curr_token = lexer.next();

        let mut to_register: Vec<(LexedToken, LexedToken, LexedToken)> = vec![];

        loop {
            if curr_token.is_none() {
                panic!("Found a none token prematurely!");
            }

            let token = curr_token.unwrap();

            let punct = lexer.next();
            let next_token = lexer.next();

            let punct = punct.unwrap();
            let next_token = next_token.unwrap();

            if next_token == LexedToken::End {
                break;
            }

            to_register.push((token, punct, next_token.clone()));

            curr_token = Some(next_token);
        }

        for (src, pct, dst) in to_register {
            let src = self.ensure_textlet_from_token(src);
            let pct = self.ensure_textlet_from_token(pct);
            let dst = self.ensure_textlet_from_token(dst);

            self.register_edge(src, dst, pct);
        }
    }
}
