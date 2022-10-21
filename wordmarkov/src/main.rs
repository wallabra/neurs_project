use rand::Rng;
use std::io;
use std::io::Write;
use wordmarkov::prelude::*;

const MAX_LEN: usize = 450;

fn parse(chain: &mut MarkovChain, prompt: &str) {
    if !prompt.is_empty() {
        chain.parse_sentence(prompt);
    }
}

fn produce(chain: &MarkovChain, prompt: &str) -> String {
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

        if words.is_empty() {
            MarkovSeed::Random
        }

        else {
            let mut rng = rand::thread_rng();
            MarkovSeed::Word(words[rng.gen_range(0..words.len())])
        }
    } else {
        MarkovSeed::Random
    };

    let res = chain.compose_sentence(seed, &mut WeightedRandomSelector, Some(MAX_LEN));

    match res {
        Ok(res) => res.to_string(),
        Err(res) => format!("{{ ERROR: {} }}", res)
    }
}

fn main() {
    let mut chain: MarkovChain = MarkovChain::new();
    let mut buffer = String::new();
    let stdin = io::stdin();

    print!("> ");
    io::stdout().flush().unwrap();

    while stdin.read_line(&mut buffer).is_ok() {
        parse(&mut chain, &buffer.trim());
        print!("{}\n\n> ", produce(&chain, &buffer.trim()));
        io::stdout().flush().unwrap();
    }
}
