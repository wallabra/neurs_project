#![cfg(test)]

use wordmarkov::prelude::*;

#[test]
fn test_split_sentence_1() {
    let sentence = "Nice tea, mate.";

    let mut lexstate = Lexer::new(sentence);

    assert_eq!(lexstate.next(), Some(Token::Begin));
    assert_eq!(lexstate.next(), Some(Token::Punct("")));
    assert_eq!(lexstate.next(), Some(Token::Word("Nice")));
    assert_eq!(lexstate.next(), Some(Token::Punct(" ")));
    assert_eq!(lexstate.next(), Some(Token::Word("tea")));
    assert_eq!(lexstate.next(), Some(Token::Punct(", ")));
    assert_eq!(lexstate.next(), Some(Token::Word("mate")));
    assert_eq!(lexstate.next(), Some(Token::Punct(".")));
    assert_eq!(lexstate.next(), Some(Token::End));
    assert_eq!(lexstate.next(), None);
    assert_eq!(lexstate.next(), None);
}

#[test]
fn test_split_sentence_2() {
    let sentence = "[ITEM] Avocado - sweet";

    let mut lexstate = Lexer::new(sentence);

    assert_eq!(lexstate.next(), Some(Token::Begin));
    assert_eq!(lexstate.next(), Some(Token::Punct("[")));
    assert_eq!(lexstate.next(), Some(Token::Word("ITEM")));
    assert_eq!(lexstate.next(), Some(Token::Punct("] ")));
    assert_eq!(lexstate.next(), Some(Token::Word("Avocado")));
    assert_eq!(lexstate.next(), Some(Token::Punct(" - ")));
    assert_eq!(lexstate.next(), Some(Token::Word("sweet")));
    assert_eq!(lexstate.next(), Some(Token::Punct("")));
    assert_eq!(lexstate.next(), Some(Token::End));
    assert_eq!(lexstate.next(), None);
    assert_eq!(lexstate.next(), None);
}
