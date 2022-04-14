#![cfg(test)]

use super::token::*;

#[test]
fn test_token_join_1() {
    let tokens: Vec<Token> = vec![
        Token::Begin,
        Token::Word("Fancy"),
        Token::Punct(" "),
        Token::Word("hat"),
        Token::Punct(","),
        Token::Punct(" "),
        Token::Word("mate"),
        Token::Punct("."),
        Token::End,
    ];

    assert_eq!(Token::recompose(&tokens), "Fancy hat, mate.");
}
