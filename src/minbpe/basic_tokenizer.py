"""Basic Tokenizer implementation."""

from typing import Literal

from minbpe.base_tokenizer import BaseTokenizer


class BasicTokenizer(BaseTokenizer):

    def __init__(self) -> None:
        super().__init__()

    # train method to learn merges and vocab from text
    def train(
        self,
        text: str,
        vocab_size: int,
        verbose: bool = False,
        mode: Literal['text', 'file'] = 'text',
    ) -> None:
        if mode == 'file':
            try:
                with open(text, encoding='utf-8') as f:
                    text = f.read()
            except FileNotFoundError as e:
                raise ValueError(f"Error reading file {text}") from e
        if vocab_size < 256:
            raise ValueError(
                f"vocab_size must be at least 256, got {vocab_size}"
            )
        num_merges: int = vocab_size - 256
        # input text preprocessing
        text_bytes: bytes = text.encode("utf-8", errors="strict")  # raw bytes
        ids: list[int] = list(text_bytes)  # list of integers in range 0..255
        # iteratively merge the most common pairs to create new tokens
        merges: dict[tuple[int, int], int] = {}  # (int, int) -> int
        vocab: dict[int, bytes] = {}  # int -> bytes
        for idx_token in range(256):
            byte_representation: bytes = bytes([idx_token])
            vocab[idx_token] = byte_representation
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats: dict[tuple[int, int], int] = self._get_stats(ids)
            # find the pair with the highest count
            most_frequent_pair: tuple[int, int] | None
            highest_count: int
            most_frequent_pair, highest_count = self._get_most_frequent_pair(
                stats
            )
            if highest_count <= 0 or most_frequent_pair is None:
                # no more pairs can be merged
                if verbose:
                    print(
                        f"No more pairs can be merged at iteration {i}. Stopping early."
                    )
                break
            pair: tuple[int, int] = most_frequent_pair
            # mint a new token: assign it the next available id
            idx: int = 256 + i
            merged_ids: list[int] = self._merge(ids, pair, idx)
            # replace all occurrences of pair in ids with idx
            ids: list[int] = merged_ids  # type: ignore[no-redef]
            # save the merge
            merges[pair] = idx
            left_token: bytes = vocab[pair[0]]
            right_token: bytes = vocab[pair[1]]
            vocab[idx] = left_token + right_token
            # prints
            if verbose:
                print(
                    f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]!r}) had {stats[pair]} occurrences"
                )
        # save class variables
        self.token_merges = merges  # used in encode()
        self.vocab = vocab  # used in decode()

    # decode method to convert token ids back to string
    def decode(self, ids: list[int]) -> str:
        # given ids (list of integers), return Python string
        text_bytes: bytes = b""
        for idx in ids:
            text_bytes += self.vocab[idx]
        text: str = text_bytes.decode("utf-8", errors="replace")
        return text

    # encode method to convert string to token ids
    def encode(self, text: str) -> list[int]:
        # given a string text, return the token ids
        text_bytes: bytes = text.encode("utf-8", errors="strict")  # raw bytes
        ids: list[int] = list(text_bytes)  # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            # get frequency stats of all adjacent pairs
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
            best_pair: tuple[int, int] | None = None
            best_priority: float | int = float("inf")
            # find the pair with the smallest merge index ---
            for p in pair_priorities:
                priority: float | int = pair_priorities[p]
                if priority < best_priority:
                    best_priority = priority
                    best_pair = p
            if best_pair is None:
                # No pairs to merge
                break
            final_pair: tuple[int, int] = best_pair
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if final_pair not in self.token_merges:
                break  # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx: int = self.token_merges[final_pair]
            merged_ids: list[int] = self._merge(ids, final_pair, idx)
            ids: list[int] = merged_ids  # type: ignore[no-redef]
        return ids
