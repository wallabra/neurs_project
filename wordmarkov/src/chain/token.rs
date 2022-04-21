use crate::sentence::token::Token as LexedToken;
use std::collections::LinkedList;
use std::fmt::Display;
use std::fmt::Formatter;
use std::rc::Rc;

/// A Markov token.
#[derive(PartialEq, Debug)]
pub enum MarkovToken<'a> {
    Begin,
    End,
    Textlet(&'a str),
}

impl<'a> MarkovToken<'a> {
    fn string_ref(&'a self) -> &'a str {
        match self {
            MarkovToken::Textlet(s) => s,
            MarkovToken::Begin => "",
            MarkovToken::End => "",
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            MarkovToken::Textlet(s) => s.is_empty(),
            MarkovToken::Begin => true,
            MarkovToken::End => true,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            MarkovToken::Textlet(s) => s.len(),
            MarkovToken::Begin => 0,
            MarkovToken::End => 0,
        }
    }
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
#[derive(PartialEq, Debug)]
pub enum MarkovTokenOwned {
    Begin,
    End,
    Textlet(Rc<str>),
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

impl<'a> From<&'a MarkovToken<'a>> for &'a str {
    fn from(token: &'a MarkovToken<'a>) -> Self {
        token.string_ref()
    }
}

/// A list of [MarkovToken]s; a sentence.
pub struct TokenList<'a>(pub LinkedList<MarkovToken<'a>>);

impl<'a> TokenList<'a> {
    pub fn iter(&self) -> std::collections::linked_list::Iter<MarkovToken> {
        self.0.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.iter().any(|x| x.len() > 0)
    }

    pub fn len(&self) -> usize {
        self.iter().map(|x| x.len()).sum()
    }
}

impl<'a> Display for TokenList<'a> {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), std::fmt::Error> {
        for x in &self.0 {
            fmt.write_str(x.string_ref())?;
        }

        Ok(())
    }
}
