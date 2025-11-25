# -----------------------------------------------------------------------------
# common test data

import os
from typing import LiteralString

import pytest
import tiktoken

from minbpe.basic_tokenizer import BasicTokenizer
from minbpe.gpt4_tokenizer import GPT4Tokenizer
from minbpe.regex_tokenizer import RegexTokenizer

# a few strings to test the tokenizers on
test_strings: list[str] = [
    "",  # empty string
    ".",  # single character
    """
    Hello こんにちは(kon'nichiwa = hello in Japanese)
    你好(nǐ hǎo = hello in Chinese) नमस्ते(namaste = hello/greetings in Hindi)
    and welcome to 2025! 🌍
    """,
    """
    I ❤️ love café au lait in Paris, пить чай(pit' chay = to drink tea in Russian) con leche, and 먹다(meokda = to eat in Korean) 김치(kimchi = kimchi) under the 서울(seoul = Seoul) skyline! ☕🇫🇷🇷🇺🇰🇷
    """,
    "FILE:taylorswift.txt",  # FILE: is handled as a special string in unpack()
    "FILE:test.txt",
]


def unpack(text: str) -> str:
    # we do this because `pytest -v .` prints the arguments to console, and we don't
    # want to print the entire contents of the file, it creates a mess. So here we go.
    if text.startswith("FILE:"):
        dirname: str = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(dirname, "..", "input")
        file_name: str = os.path.join(input_dir, text[5:])
        contents: str = open(file_name, encoding="utf-8").read()
        return contents
    else:
        return text


# special tokens test string
specials_string: LiteralString = (
    """
    <|endoftext|>HThis is first document, sometisdsfdsdsfsdsfdsfsdfdsf
    <|endoftext|>And this is another document, some more text here.
    <|endoftext|><|fim_prefix|>Third document, this one with <|fim_suffix|> tokens.<|fim_middle|> FIM
    <|endoftext|>Last document!!! 👋<|endofprompt|>
    """.strip()
)

# special tokens mapping
special_tokens: dict[str, int] = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276,
}

# complex text with special tokens
llama_text: LiteralString = (
    """
    <|endoftext|>The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.
    Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4]
    The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5]
    <|fim_prefix|>In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology,<|fim_suffix|> where they come from at the end of time.[6]<|fim_middle|> llamas will return to the water springs and ponds<|endofprompt|>
    """.strip()
)

# -----------------------------------------------------------------------------
# tests

# test encode/decode identity for a few different strings

# Pre-instantiated tokenizer objects
tokenizers: list[BasicTokenizer | GPT4Tokenizer | RegexTokenizer] = [
    BasicTokenizer(),
    RegexTokenizer(),
    GPT4Tokenizer(),
]


@pytest.mark.parametrize(
    "tokenizer", tokenizers, ids=["basic", "regex", "gpt4"]
)
@pytest.mark.parametrize("text", test_strings)
def test_encode_decode_identity(
    tokenizer: BasicTokenizer | GPT4Tokenizer | RegexTokenizer, text: str
) -> None:
    text_unpacked: str = unpack(text)
    ids: list[int] = tokenizer.encode(text_unpacked)
    decoded: str = tokenizer.decode(ids)
    if text_unpacked != decoded:
        raise AssertionError(
            f"Tokenizer failed identity check.\n"
            f"Original: {text_unpacked}\n"
            f"Decoded:  {decoded}"
        )


# test that our tokenizer matches the official GPT-4 tokenizer
@pytest.mark.parametrize("text", test_strings)
def test_gpt4_tiktoken_equality(text: str) -> None:
    text_unpacked: str = unpack(text)
    tokenizer: GPT4Tokenizer = GPT4Tokenizer()
    enc: tiktoken.Encoding = tiktoken.get_encoding("cl100k_base")
    tiktoken_ids: list[int] = enc.encode(text_unpacked)
    gpt4_tokenizer_ids: list[int] = tokenizer.encode(text_unpacked)
    if gpt4_tokenizer_ids != tiktoken_ids:
        raise AssertionError(
            f"GPT4 Tokinezer did not return the same as TikToken.\n"
            f"Original: {text}\n"
        )


# test the handling of special tokens
def test_gpt4_tiktoken_equality_special_tokens() -> None:
    tokenizer: GPT4Tokenizer = GPT4Tokenizer()
    enc: tiktoken.Encoding = tiktoken.get_encoding("cl100k_base")
    tiktoken_ids: list[int] = enc.encode(
        specials_string, allowed_special="all"
    )
    gpt4_tokenizer_ids: list[int] = tokenizer.encode(
        specials_string, allowed_special="all"
    )
    if gpt4_tokenizer_ids != tiktoken_ids:
        raise AssertionError(
            f"GPT4 Tokenizer did not handle special tokens correctly.\n"
            f"Original: {specials_string}\n"
        )


tokenizers_test: list[BasicTokenizer | RegexTokenizer] = [
    BasicTokenizer(),
    RegexTokenizer(),
]


# basic train test
@pytest.mark.parametrize(
    "tokenizers_train", tokenizers_test, ids=["basic", "regex"]
)
def test_wikipedia_example(
    tokenizers_train: BasicTokenizer | RegexTokenizer,
) -> None:
    """
    Quick unit test, following along the Wikipedia example:
    https://en.wikipedia.org/wiki/Byte_pair_encoding

    According to Wikipedia, running bpe on the input string:
    "aaabdaaabac"

    for 3 merges will result in string:
    "XdXac"

    where:
    X=ZY
    Y=ab
    Z=aa

    Keep in mind that for us a=97, b=98, c=99, d=100 (ASCII values)
    so Z will be 256, Y will be 257, X will be 258.

    So we expect the output list of ids to be [258, 100, 258, 97, 99]
    """
    tokenizer: BasicTokenizer | RegexTokenizer = tokenizers_train
    text: str = "aaabdaaabac"
    tokenizer.train(text, 256 + 3)
    ids: list[int] = tokenizer.encode(text)
    if ids != [258, 100, 258, 97, 99]:
        raise AssertionError(
            f"Tokenizer did not produce expected output.\n"
            f"Original: {text}\n"
            f"Encoded:  {ids}"
        )
    if tokenizer.decode(tokenizer.encode(text)) != text:
        raise AssertionError(
            f"Tokenizer failed identity check after training.\n"
            f"Original: {text}\n"
            f"Decoded:  {tokenizer.decode(tokenizer.encode(text))}"
        )


@pytest.mark.parametrize(
    "special_tokens", [{}, special_tokens], ids=["none", "with_specials"]
)
def test_save_load(special_tokens: dict[str, int]) -> None:
    # take a bit more complex piece of text and train the tokenizer, chosen at random
    text: str = llama_text
    # create a Tokenizer and do 64 merges
    tokenizer: RegexTokenizer = RegexTokenizer()
    tokenizer.train(text, 256 + 64)
    tokenizer.register_special_tokens(special_tokens)
    # verify that decode(encode(x)) == x
    if tokenizer.decode(tokenizer.encode(text, "all")) != text:
        raise AssertionError(
            f"Tokenizer failed decoding text to the original that was used to train.\n"
            f"Original: {text}\n"
            f"Decoded:  {tokenizer.decode(tokenizer.encode(text, 'all'))}"
        )
    # verify that save/load work as expected
    ids: list[int] = tokenizer.encode(text, "all")
    # save the tokenizer
    tokenizer.save("src/output/", "test_tokenizer_tmp")
    # re-load the tokenizer
    new_tokenizer: RegexTokenizer = RegexTokenizer()
    new_tokenizer.load("src/output/", "test_tokenizer_tmp.model")
    # verify that decode(encode(x)) == x
    if new_tokenizer.decode(ids) != text:
        raise AssertionError(
            f"New Loaded Tokenizer failed decoding text to the original that was used to train.\n"
            f"Original: {text}\n"
            f"Decoded:  {new_tokenizer.decode(ids)}"
        )
    if new_tokenizer.decode(new_tokenizer.encode(text, "all")) != text:
        raise AssertionError(
            f"New Loaded Tokenizer failed identity check after loading, encode/decode conflict.\n"
            f"Original: {text}\n"
            f"Decoded:  {new_tokenizer.decode(new_tokenizer.encode(text, 'all'))}"
        )
    if new_tokenizer.encode(text, "all") != ids:
        raise AssertionError(
            f" New Loaded Tokenizer failed encoding did not match previous encode.\n"
            f"Original: {text}\n"
            f"Old IDs:  {ids}\n"
            f"New IDs:  {tokenizer.encode(text, 'all')}"
        )
    # delete the temporary files
    for file in [
        "src/output/test_tokenizer_tmp.model",
        "src/output/test_tokenizer_tmp.vocab",
    ]:
        os.remove(file)


if __name__ == "__main__":
    pytest.main()
