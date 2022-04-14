#![cfg(test)]

use super::lex::Lexer;
use super::token::Token;

#[test]
fn test_split_sentence() {
    let sentence = "Nice tea, mate.";

    let mut lexstate = Lexer::new(sentence);

    assert_eq!(lexstate.next(), Some(Token::Begin));
    assert_eq!(lexstate.next(), Some(Token::Word("Nice")));
    assert_eq!(lexstate.next(), Some(Token::White(" ")));
    assert_eq!(lexstate.next(), Some(Token::Word("tea")));
    assert_eq!(lexstate.next(), Some(Token::Punct(",")));
    assert_eq!(lexstate.next(), Some(Token::White(" ")));
    assert_eq!(lexstate.next(), Some(Token::Word("mate")));
    assert_eq!(lexstate.next(), Some(Token::Punct(".")));
    assert_eq!(lexstate.next(), Some(Token::End));
    assert_eq!(lexstate.next(), None);
    assert_eq!(lexstate.next(), None);
}
