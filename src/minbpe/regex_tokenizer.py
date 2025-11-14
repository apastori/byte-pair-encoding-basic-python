"""Regex Tokenizer implementation."""

import regex
from typing import Final, Literal
from minbpe.base_tokenizer import BaseTokenizer
from minbpe.const_protector import ConstProtector

class RegexTokenizer(BaseTokenizer, metaclass=ConstProtector):
     
     # Constants (by convention, uppercase = constant)
    _GPT2_SPLIT_PATTERN: Final[str] = (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    _GPT4_SPLIT_PATTERN: Final[str] = (
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    )

    def __init__(self, pattern: str = None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
            example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern: str = (
            RegexTokenizer._GPT4_SPLIT_PATTERN if pattern is None else pattern
        )
        self.compiled_pattern: regex.Pattern = regex.compile(self.pattern)
        # special_tokens is already defined in BaseTokenizer, but we
        # initialize it here to an empty dict for clarity
        # special_tokens is for encoding
        self.special_tokens: dict[str, int] = {}
        # inverse_special_tokens is for decoding
        self.inverse_special_tokens: dict[int, str] = {}

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be at least 256, got {vocab_size}")
        num_merges: int = vocab_size - 256

        # split the text up into text chunks
        text_chunks: list[str] = regex.findall(self.compiled_pattern, text)
        # input text preprocessing
        ids: list[list[int]] = [] # list of list of integers in range 0..255
        for chunk in text_chunks:
            # Convert each chunk (string) into a UTF-8 encoded bytes object
            chunk_bytes: bytes = chunk.encode("utf-8", errors="strict") # raw bytes
            # Convert the bytes object into a list of integer byte values (0–255)
            chunk_ids: list[int] = list(chunk_bytes)
            # Add the result to the main list
            ids.append(chunk_ids)
        # iteratively merge the most common pairs to create new tokens
        merges: dict[tuple[int, int], int] = {} # (int, int) -> int
        vocab: dict[int, bytes] = {} # int -> bytes
        for idx in range(256):
            byte_representation: bytes = bytes([idx])
            vocab[idx] = byte_representation
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats: dict[tuple[int, int], int] = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                chunk_stats: dict[tuple[int, int], int] = self._get_stats(chunk_ids)
                # Merge the chunk's stats into the global stats dictionary
                for pair, count in chunk_stats.items():
                    # If this pair already exists in the global dictionary, add to its count
                    if pair in stats:
                        stats[pair] += count
                    # Otherwise, start counting this pair from this chunk's count
                    else:
                        stats[pair] = count
            # find the pair with the highest count
            most_frequent_pair: tuple[int, int] | None
            highest_count: int
            most_frequent_pair, highest_count = self._get_most_frequent_pair(stats)
            if highest_count <= 0 or most_frequent_pair is None:
                # no more pairs can be merged
                if verbose:
                    print(f"No more pairs can be merged at iteration {i}. Stopping early.")
                break
            pair: tuple[int, int] = most_frequent_pair
            # mint a new token: assign it the next available id
            idx: int = 256 + i
            # replace all occurrences of pair in ids with idx
            # For every list of token IDs (chunk) in ids
            updated_ids: list[list[int]] = []
            for chunk_ids in ids:
                # Merge all occurrences of the current pair into a new token ID
                merged_chunk: list[int] = self._merge(chunk_ids, pair, idx)
                # Add the merged chunk to the updated list
                updated_ids.append(merged_chunk)
            # Replace the old ids with the updated ones
            ids = updated_ids
            # save the merge
            merges[pair] = idx
            left_token: bytes = vocab[pair[0]]
            right_token: bytes = vocab[pair[1]]
            vocab[idx] = left_token + right_token
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.token_merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        """
        Register special tokens and their corresponding IDs.

        Args:
            special_tokens (dict[str, int]): A mapping of token strings to integer IDs.
                Example:
                    {
                        "<|endoftext|>": 100257,
                        "<|pad|>": 100258
                    }

        This method also creates an inverse mapping (int -> str) 
        to allow decoding IDs back into their special token strings.
        """
        self.special_tokens = special_tokens
        # Initialize an empty dictionary for the inverse mapping
        inv_special_tokens: dict[int, str] = {}
        # Loop through each token and its ID
        for token_str, token_id in special_tokens.items():
            # Add the reversed mapping: token_id -> token_str
            inv_special_tokens[token_id] = token_str
        self.inverse_special_tokens = inv_special_tokens

    def decode(self, ids: list[int]) -> str:
        # given ids (list of integers), return Python string
        part_bytes: list[bytes] = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                special_token_str: str = self.inverse_special_tokens[idx]
                part_bytes.append(special_token_str.encode("utf-8", errors="strict"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes: bytes = b"".join(part_bytes)
        text: str = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def _encode_chunk(self, text_bytes: bytes) -> list[int]:
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids: list[int] = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats: dict[tuple[int, int], int] = self._get_stats(ids)
            # collect all current pairs ---
            pairs_list: list[tuple[int, int]] = list(stats.keys())
            # assign each pair its merge priority ---
            pair_priorities: dict[tuple[int, int], float | int] = {}
            for p in pairs_list:
                if p in self.token_merges:
                    pair_priorities[p] = self.token_merges[p]
                else:
                    pair_priorities[p] = float("inf")
            # initialize variables to track the best pair and its smallest priority
            best_pair: tuple[int, int] = None
            best_priority: float | int = float("inf")
            # find the pair with the smallest merge index ---
            for p in pair_priorities:
                priority: float | int = pair_priorities[p]
                if priority < best_priority:
                    best_priority = priority
                    best_pair = p
            pair: tuple[int, int] = best_pair
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.token_merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx: int = self.token_merges[pair]
            ids: list[int] = self._merge(ids, pair, idx)
        return ids
    
    def encode_ordinary(self, text: str) -> list[int]:
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks: list[str] = regex.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids: list[int] = []
        for chunk in text_chunks:
            chunk_bytes: bytes = chunk.encode("utf-8", errors="strict") # raw bytes
            chunk_ids: list[int] = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
    
    def encode(self, text: str, allowed_special: Literal["all", "none", "none_raise"] | set[str] = "none_raise") -> list[int]:
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special: dict[str, int] = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            for token in self.special_tokens:
                if token in text:
                    raise ValueError(f"special token {token} found in text, but allowed_special='none_raise'")
        elif isinstance(allowed_special, set):
            # Iterate over each key (k) and value (v) in all available special tokens
            for k_special_token, v_special_token in self.special_tokens.items():
                # Check if the key (e.g., "<|endoftext|>", "<|user|>", etc.)
                # is present in the user-provided 'allowed_special' set
                if k_special_token in allowed_special:
                    special[k_special_token] = v_special_token
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        # Create an empty list to hold the escaped special tokens
        escaped_tokens: list[str] = []
        for key_special in special:
            # This ensures characters like '|' are treated as literal text, not regex commands.
            escaped_k: str = regex.escape(key_special)
            escaped_tokens.append(escaped_k)
        # Join all the escaped tokens together with the "|" (OR) operator
        # This builds the core of the pattern: "<\|user\|>|<\|bot\|>|..."
        inner_pattern: str = "|".join(escaped_tokens)
        special_pattern: str = "(" + inner_pattern + ")"
        special_chunks: list[str] = regex.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids: list[int] = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids