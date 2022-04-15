/*!
 * Code for splitting a sentence string into [super::token::Token]s.
 */
pub use super::token::Token;

#[derive(PartialEq, Debug)]
enum LexingType {
    Begin,
    White,
    Punct,
    Word,
    End,
    Empty,
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
            LexingType::White => Token::White(s),
            LexingType::Punct => Token::Punct(s),
            LexingType::Word => Token::Word(s),
            LexingType::Begin => Token::Begin,
            LexingType::End => Token::End,
            LexingType::Empty => {
                panic!("Internal code error: tried to use state_wrap on an exhausted LexingState!")
            }
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
            return LexingType::End;
        }

        let char = char.unwrap();

        if char.is_ascii_punctuation() {
            LexingType::Punct
        } else if char.is_whitespace() {
            LexingType::White
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

        if LexingType::End == self.state {
            self.state = LexingType::Empty;
            return Some(Token::End);
        }

        let mut chars = self.from.chars().skip(self.head);

        loop {
            let char = chars.next();
            let ctype = self.char_type(char);

            if cfg!(test) {
                println!(
                    "{}..={}\t {:?}\t | {:?} \t{:?}\t ({:?})",
                    self.start,
                    self.head,
                    self.state,
                    char,
                    ctype,
                    self.peek_next()
                );
            }

            if ctype != self.state {
                let res = self.peek_next();
                self.state = ctype;

                if res != Token::Begin {
                    self.start = self.head;
                }

                return Some(res);
            }

            self.head += 1;
        }
    }
}
