/*!
 * Code for the synthesis and representation of sentence tokens.
 *
 * Words, spacing and punctuation are all fundamental parts of a sentence.
 */

/**
 * A token – can be either a Word or a Punct, or the beginning or end of a
 * parsed sentence.
 */
#[derive(PartialEq, Debug)]
pub enum Token<'a> {
    /// A word, a sequence of characters bounded by non-words.
    ///
    /// A non-word in this case referring to either [Token::Punct],
    /// [Token::Begin] or [Token::End].
    Word(&'a str),

    /// Whitespace.
    White(&'a str),

    /// Punctuation.
    Punct(&'a str),

    /// The beginning of a sentence.
    Begin,

    /// The end of a sentence.
    End,
}

impl<'a> Token<'a> {
    /// Recompose a slice or list of Tokens into a String.
    pub fn recompose(tokens: &'a [Token<'a>]) -> String {
        tokens
            .iter()
            .map(|x: &'a Token| -> String {
                let s: &'a str = x.into();
                s.into()
            })
            .collect()
    }
}

impl<'a> Into<&'a str> for &Token<'a> {
    /// Converts this token into the equivalent string.
    fn into(self) -> &'a str {
        match self {
            Token::Word(s) => s,
            Token::Punct(s) => s,
            Token::White(s) => s,
            Token::Begin => "",
            Token::End => "",
        }
    }
}
