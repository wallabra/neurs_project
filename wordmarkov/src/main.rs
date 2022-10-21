use rand::Rng;
use std::io;
use std::io::Write;
use wordmarkov::prelude::*;

const MAX_LEN: usize = 450;

fn parse(chain: &mut MarkovChain, prompt: &String) {
    if !prompt.is_empty() {
        chain.parse_sentence(prompt);
    }
}

fn produce(chain: &MarkovChain, prompt: &String) -> String {
    let seed = if !prompt.is_empty() {
        let lexed = Lexer::new(prompt);
        let words: Vec<&str> = lexed
            .filter_map(|lex| {
                if let Token::Word(w) = lex {
                    Some(w)
                } else {
                    None
                }
            })
            .collect();

        let mut rng = rand::thread_rng();
        MarkovSeed::Word(words[rng.gen_range(0..words.len())])
    } else {
        MarkovSeed::Random
    };

    chain
        .compose_sentence(seed, &mut WeightedRandomSelector, Some(MAX_LEN))
        .unwrap()
        .to_string()
}

fn main() {
    let mut chain: MarkovChain = MarkovChain::new();
    let mut buffer = String::new();
    let stdin = io::stdin();

    print!("> ");
    io::stdout().flush().unwrap();

    while stdin.read_line(&mut buffer).is_ok() {
        parse(&mut chain, &buffer);
        print!("\n{}\n\n> ", produce(&chain, &buffer));
        io::stdout().flush().unwrap();
    }
}
