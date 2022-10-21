/*!
 * Code for splitting a sentence string into [super::token::Token]s.
 */
pub use super::token::Token;

#[derive(PartialEq, Debug)]
enum LexingType {
    Begin,
    Punct,
    Word,
    End,
    Empty,

    PreEnd,
    PostBegin,
}

/**
 * A structure that allows splitting a sentence into [Token]s.
 */
pub struct Lexer<'a> {
    from: &'a str,
    start: usize,
    head: usize,
    state: LexingType,
}

impl<'a> Lexer<'a> {
    /**
     * Make a new Lexer state from a string.
     */
    pub fn new(from: &'a str) -> Lexer<'a> {
        Lexer {
            from,
            start: 0,
            head: 0,
            state: LexingType::Begin,
        }
    }

    fn state_wrap(&self, s: &'a str) -> Token<'a> {
        match self.state {
            LexingType::Punct => Token::Punct(s),
            LexingType::Word => Token::Word(s),
            LexingType::Begin => Token::Begin,
            LexingType::End => Token::End,
            LexingType::Empty => {
                panic!("Internal code error: tried to use state_wrap on an exhausted LexingState!")
            }

            // special cases
            LexingType::PostBegin => Token::Punct(s),
            LexingType::PreEnd => Token::Punct(s),
        }
    }

    fn peek_next(&self) -> Token<'a> {
        if self.state == LexingType::Begin {
            return Token::Begin;
        }

        if self.state == LexingType::End {
            return Token::End;
        }

        self.state_wrap(&self.from[self.start..self.head])
    }

    fn char_type(&self, char: Option<char>) -> LexingType {
        if char.is_none() {
            if self.state == LexingType::Punct {
                return LexingType::End;
            }

            return LexingType::PreEnd;
        }

        let char = char.unwrap();

        if char.is_ascii_punctuation() || char.is_whitespace() {
            LexingType::Punct
        } else {
            LexingType::Word
        }
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Token<'a>> {
        if LexingType::Empty == self.state {
            return None;
        }

        if LexingType::Begin == self.state {
            self.state = LexingType::PostBegin;
            return Some(Token::Begin);
        }

        if LexingType::End == self.state {
            self.state = LexingType::Empty;
            return Some(Token::End);
        }

        if LexingType::PreEnd == self.state {
            self.state = LexingType::End;
            return Some(Token::Punct(""));
        }

        let chars = &mut self.from[self.head..].chars();

        loop {
            let nextchar = chars.next();
            let ctype = self.char_type(nextchar);

            let final_ctype = if self.state == LexingType::PostBegin {
                &LexingType::Punct
            } else {
                &self.state
            };

            if &ctype != final_ctype {
                let res = self.peek_next();

                self.state = ctype;

                if res != Token::Begin {
                    self.start = self.head;
                }

                return Some(res);
            }

            self.head += nextchar.map_or(1, |x| x.len_utf8());
        }
    }
}
