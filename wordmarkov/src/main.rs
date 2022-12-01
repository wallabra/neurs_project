use rand::Rng;
use std::io::{self, BufRead, Write};
use std::{env, fs};
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
        } else {
            let mut rng = rand::thread_rng();
            MarkovSeed::Word(words[rng.gen_range(0..words.len())])
        }
    } else {
        MarkovSeed::Random
    };

    let res = chain.compose_sentence(seed, &mut WeightedRandomSelector, Some(MAX_LEN));

    match res {
        Ok(res) => res.to_string(),
        Err(res) => format!("{{ ERROR: {} }}", res),
    }
}

fn parse_file(chain: &mut MarkovChain, path: &str) -> io::Result<()> {
    let file = fs::File::open(path)?;

    for line in io::BufReader::new(file).lines().flatten() {
        parse(chain, line.trim());
    }

    Ok(())
}

fn status_line(chain: &MarkovChain) -> String {
    format!("t{} e{}", chain.len(), chain.num_edges())
}

fn main() {
    let mut chain: MarkovChain = MarkovChain::new();

    // Read files from command args to parse into the chain.
    let args: Vec<String> = env::args().collect();

    for arg in &args[1..] {
        if let Err(err) = parse_file(&mut chain, arg) {
            println!("WARN: Error reading file {}: {}", arg, err);
        }
    }

    // Start the prompt loop.
    let mut buffer = String::new();
    let stdin = io::stdin();
 
    print!("({})> ", status_line(&chain));
    io::stdout().flush().unwrap();

    while stdin.read_line(&mut buffer).is_ok() {
        let trimmed = buffer.trim();
        let learn = matches!(trimmed.chars().next(), Some(':'));

        let trimmed = if learn { &trimmed[1..] } else { trimmed };

        print!("{}\n\n", produce(&chain, trimmed));

        if learn {
            parse(&mut chain, trimmed);
        }

        print!("({})> ", status_line(&chain));
        io::stdout().flush().unwrap();
        buffer.clear();
    }
}
