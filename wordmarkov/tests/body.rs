#![cfg(test)]

use wordmarkov::prelude::*;

#[test]
fn test_chain_parsing() {
    let mut chain: MarkovChain = MarkovChain::new();

    chain.parse_sentence("Mary had a little lamb");

    assert_eq!(chain.len(), 9); // BEGIN and END count, as well as ' ' and ''!
    assert_eq!(chain.num_edges(), 6);
    assert_eq!(chain.num_textlets(), 9);

    // Order usually doesn't really matter, but psst,
    // if it's well-defined behaviour, might as well test
    // it. :)
    assert_eq!(chain.try_get_textlet_index(""), Some(2));
    assert_eq!(chain.try_get_textlet_index(" "), Some(4));
    assert_eq!(chain.try_get_textlet_index("."), None);
}

#[test]
fn test_chain_traversal() {
    let mut chain: MarkovChain = MarkovChain::new();

    chain.parse_sentence(
        "a lamb ate a lamb made a lamb wear a little lamb with a lamb on top of that one lamb who lambed over lamb with a cute lamb",
    );

    let max_len = 500;

    let new_sentence = chain
        .compose_sentence(
            MarkovSeed::Word("lamb"),
            &mut WeightedRandomSelector,
            Some(max_len),
        )
        .unwrap();

    assert!(new_sentence.len() < max_len);
    assert!(new_sentence.to_string().contains("lamb"));

    println!("Composed sentence: {}", new_sentence);
}
