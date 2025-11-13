""" Small GPT-4 style wrapper on the Regex implementation."""

from types import MappingProxyType

import tiktoken
from minbpe.const_protector import ConstProtector
from minbpe.regex_tokenizer import RegexTokenizer
from typing import Final

class GPT4Tokenizer(RegexTokenizer, metaclass=ConstProtector):
    """GPT-4 style tokenizer implementation based on RegexTokenizer.

    This class uses special tokens for encoding and decoding, preventing
    modification of these constants via the ConstProtector metaclass.
    """

    _GPT4_SPLIT_PATTERN: Final[str] = (
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    )

    _GPT4_SPECIAL_TOKENS: Final[dict[str, int]] = MappingProxyType({
        '<|endoftext|>': 100257,
        '<|fim_prefix|>': 100258,
        '<|fim_middle|>': 100259,
        '<|fim_suffix|>': 100260,
        '<|endofprompt|>': 100276
    })

    def __init__(self):
        super().__init__(pattern=GPT4Tokenizer._GPT4_SPLIT_PATTERN)
        # get the official tokenizer and its merges
        enc: tiktoken.Encoding = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks: dict[bytes, int] = enc._mergeable_ranks
        self.token_merges = self._recover_merges(mergeable_ranks)
        # build the vocabulary
        self.vocab = self._build_vocab()
        # build byte shuffle mapping due to historical error in tiktoken
        self.byte_shuffle = self._build_byte_shuffle(mergeable_ranks)
        # build inverse byte shuffle mapping due to historical error in tiktoken
        self.inverse_byte_shuffle = self._build_inverse_byte_shuffle()
        # register special tokens
        self.register_special_tokens(GPT4Tokenizer._GPT4_SPECIAL_TOKENS)

    # Private method to build the vocabulary
    def _build_vocab(self) -> dict[int, bytes]:
        # Create an empty dictionary to store byte mappings
        vocab: dict[int, bytes] = {}
        # Loop through all possible byte values (0 to 255)
        for byte_value in range(256):
            # Convert the integer to a single-byte bytes object
            byte_representation = bytes([byte_value])
            # Map the integer (key) to its bytes object (value)
            vocab[byte_value] = byte_representation
        # loop through token merges to build combined tokens
        for (p0, p1), idx in self.token_merges.items():
            # Ensure inputs are bytes (if vocab might have mixed types)
            p0_bytes = vocab[p0]
            p1_bytes = vocab[p1]
            vocab[idx] = p0_bytes + p1_bytes
        # Add special tokens to the vocabulary
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode(encoding='utf-8', errors='strict')
        return vocab
    
    # create byte shuffle mapping for correct order single byte,
    # historical error with tiktoken with the ascii order of bytes
    def _build_byte_shuffle(self, ranks_tiktoken: dict[bytes, int]) -> dict[int, int]:
        byte_shuffle: dict[int, int] = {}
        for i in range(256):
            byte_representation: bytes = bytes([i])
            byte_shuffle[i] = ranks_tiktoken[byte_representation]
        return byte_shuffle
    
    # create inverse shuffle mapping for correct order single byte,
    # historical error with tiktoken with the ascii order of bytes
    def _build_inverse_byte_shuffle(self) -> dict[int, int]:
        inverse_byte_shuffle: dict[int, int] = {}
        for k_byte_shuffle, v_byte_shuffle in self.byte_shuffle.items():
            self.inverse_byte_shuffle[v_byte_shuffle] = k_byte_shuffle
        return inverse_byte_shuffle

    def decode(self, ids: list[int]) -> str:
        # we have to un-permute the bytes before we decode
        # Step 1: reconstruct the byte sequence from vocab
        text_bytes_parts: list[bytes] = []
        for idx in ids:
            vocab_entry: bytes = self.vocab[idx]
            text_bytes_parts.append(vocab_entry)
        text_bytes: bytes = b"".join(text_bytes_parts)
         # Step 2: apply inverse byte shuffle
        shuffled_bytes: list[int] = []
        for b in text_bytes:
            inverse_b: int = self.inverse_byte_shuffle[b]
            shuffled_bytes.append(inverse_b)
        text_bytes_final: bytes = bytes(shuffled_bytes)
        text: str = text_bytes_final.decode("utf-8", errors="replace")
        return text

    # this is a pretrained tokenizer, it is not intended to be trained
    def train(self, text: str, vocab_size: int, verbose=False) -> None:
        raise NotImplementedError
    
    def _encode_chunk(self, text_bytes: bytes) -> list[int]:
        # before we start processing bytes, we have to permute them
        # Step 1: apply byte shuffle permutation
        shuffled_bytes_list: list[int] = []
        for b in text_bytes:
            shuffled_value: int = self.byte_shuffle[b]
            shuffled_bytes_list.append(shuffled_value)
        text_bytes: bytes = bytes(shuffled_bytes_list)
        ids: list[int] = super()._encode_chunk(text_bytes)
        return ids

    def _recover_merges(self, mergeable_ranks: dict[bytes, int]) -> dict[tuple[int, int], int]:
        # the `merges` are already the byte sequences in their merged state.
        # so we have to recover the original pairings. We can do this by doing
        # a small BPE training run on all the tokens, in their order.
        # also see https://github.com/openai/tiktoken/issues/60
        # also see https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306
        merges: dict[tuple[int, int], int] = {}
        for token, rank in mergeable_ranks.items():
            if len(token) == 1:
                # skip raw bytes, this is only for merged tokens
                continue
            bpe_pair: list[bytes] = self._byte_pair_encoding(mergeable_ranks, token, max_rank=rank)
            if len(bpe_pair) != 2:
                raise ValueError(f"Expected pair to have exactly 2 elements, got {len(pair)}")
            pair: tuple[bytes, bytes] = tuple(bpe_pair)
            # recover the integer ranks of the pair
            ix0: int = mergeable_ranks[pair[0]]
            ix1: int = mergeable_ranks[pair[1]]
            ranks: tuple[int, int] = (ix0, ix1)
            merges[ranks] = rank
        return merges
    
    def _byte_pair_encoding(self, mergeable_ranks: dict[bytes, int], token: bytes, max_rank: int | None = None) -> list[bytes]:
        """
        Reconstructs how a byte token would be split (or merged) according to
        the Byte Pair Encoding (BPE) ranks.

        Args:
            mergeable_ranks: Mapping from byte sequences to integer ranks.
            token: A single token as a bytes object (e.g. b'hello').
            max_rank: Optional cutoff; merges with rank >= max_rank are skipped.

        Returns:
            A list of bytes objects representing the token split into
            mergeable subparts.
        """
        parts: list[bytes] = []
        for parts_byte in token:
            parts_byte_representation: bytes = bytes([parts_byte])
            parts.append(parts_byte_representation)
        while True:
            min_idx: int | None = None
            min_rank: int | None = None
            for i in range(len(parts) - 1):
                left: bytes = parts[i]
                right: bytes = parts[i + 1]
                pair_bytes: bytes = left + right
                rank = mergeable_ranks.get(pair_bytes)
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank
            if min_rank is None or (max_rank is not None and min_rank >= max_rank):
                break
            if min_idx is None:
                raise ValueError("Expected min_idx to be set, got None")    
            parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
        return parts
    


