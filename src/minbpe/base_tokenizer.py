"""Abstract Base class for Tokenizers."""

import os
import unicodedata
from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    """Abstract Base class for Tokenizers."""

    def __init__(self) -> None:
        """Initialize the base abstract tokenizer with default attributes."""
        # default: vocab size of 256 (all bytes), no token_merges, no source
        self.token_merges: dict[tuple[int, int], int] = {}
        self.source: str = ""
        self.special_tokens: dict[str, int] = {}  # e.g. {'\u0000': 100257}
        self.vocab: dict[int, bytes] = self._build_vocab()

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

    @abstractmethod
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train the tokenizer vocabulary from text."""
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token IDs."""
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back into text."""
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError

    # Private method to get frequencies of adjacent token pairs
    def _get_stats(self, ids: list[int]) -> dict[tuple[int, int], int]:
        freq_two_tokens: dict[tuple[int, int], int] = (
            {}
        )  # Empty dictionary to store pairs and their counts
        # Loop through the list from the first to the second-to-last item
        for i in range(len(ids) - 1):
            # Create a pair of current and next item
            pair: tuple[int, int] = (ids[i], ids[i + 1])
            # Check if the pair is already in the dictionary
            if pair in freq_two_tokens:
                freq_two_tokens[pair] = (
                    freq_two_tokens[pair] + 1
                )  # Increment the count
            else:
                freq_two_tokens[pair] = 1  # Start the count at 1
        return freq_two_tokens  # Return the dictionary of counts

    # Private method to merge pairs in the list of token ids
    def _merge(
        self, ids: list[int], pair: tuple[int, int], new_token: int
    ) -> list[int]:
        """In the list of integers (ids), replace all consecutive occurrences
        of pair with the new integer token idx
        Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
        """
        merge: list[int] = []
        i: int = 0
        while i < len(ids):
            # if not at the very last position AND the pair matches, replace it
            if i < len(ids) - 1:
                current, next_token = ids[i], ids[i + 1]
                # If we found the target pair, replace it
                if (current, next_token) == pair:
                    merge.append(new_token)
                    i += 2  # skip both tokens
                    continue
            # Otherwise, just keep the current token
            merge.append(ids[i])
            i += 1
        return merge

    def _replace_control_characters(self, s: str) -> str:
        """Replace control characters in a string with their Unicode escape sequences."""
        # we don't want to print control characters
        # which distort the output (e.g. \n or much worse)
        # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
        # http://www.unicode.org/reports/tr44/#GC_Values_Table
        chars: list[str] = []
        for ch in s:
            category: str = unicodedata.category(ch)
            if category[0] != "C":
                chars.append(ch)  # this character is ok
                continue
            # Only reach here for control characters
            # Unicode integer value in 4-digit hex format
            code_point: int = ord(ch)
            escaped: str = f"\\u{code_point:04x}"
            chars.append(escaped)  # escape
        return "".join(chars)

    def _render_token(self, token: bytes) -> str:
        """Decode a bytes token into a readable string,
        replacing invalid UTF-8 bytes and escaping control characters.
        """
        decoded: str = token.decode("utf-8", errors="replace")
        safe_string: str = self._replace_control_characters(decoded)
        return safe_string

    def save(self, save_dir: str, file_prefix: str) -> None:
        """Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # --- Sanity check ---
        if not os.path.isdir(save_dir):
            raise FileNotFoundError(f"Directory does not exist: {save_dir}")
        # write the model: to be used in load() later
        self._save_model_file(save_dir, file_prefix)
        self._save_vocab_file(save_dir, file_prefix)

    def _save_model_file(self, save_dir: str, file_prefix: str) -> None:
        """Save the model file (critical for load())."""
        # --- Sanity check ---
        if not os.path.isdir(save_dir):
            raise FileNotFoundError(f"Directory does not exist: {save_dir}")
        model_file_prefix: str = file_prefix + ".model"
        model_file: str = os.path.join(save_dir, model_file_prefix)
        with open(model_file, 'w', encoding='utf-8') as f:
            f.write("minbpe v1\n")
            f.write(f"{self.source}\n")
            # write special tokens
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # write token merges
            for idx1, idx2 in self.token_merges:
                f.write(f"{idx1} {idx2}\n")

    def _save_vocab_file(self, save_dir: str, file_prefix: str) -> None:
        """Save the vocab file (for human inspection only)."""
        # --- Sanity check ---
        if not os.path.isdir(save_dir):
            raise FileNotFoundError(f"Directory does not exist: {save_dir}")
        vocab_file_prefix: str = file_prefix + ".vocab"
        vocab_file: str = os.path.join(save_dir, vocab_file_prefix)
        inverted_merges: dict[int, tuple[int, int]] = {}
        for pair, new_idx in self.token_merges.items():
            inverted_merges[new_idx] = pair
        with open(vocab_file, "w", encoding="utf-8") as f:
            for token_idx, token_bytes in self.vocab.items():
                token_string: str = self._render_token(
                    token_bytes
                )  # assumes render_token() is accessible
                if token_idx in inverted_merges:
                    # token has children: render as merge
                    left_idx, right_idx = inverted_merges[token_idx]
                    left_string: str = self._render_token(self.vocab[left_idx])
                    right_string: str = self._render_token(
                        self.vocab[right_idx]
                    )
                    f.write(
                        f"[{left_string}][{right_string}] -> [{token_string}] {token_idx}\n"
                    )
                else:
                    # leaf token
                    f.write(f"[{token_string}] {token_idx}\n")

    def load(self, model_path: str, model_filename: str) -> None:
        """Load the values of model file to the tokenizer"""
        model_full_path: str = os.path.join(model_path, model_filename)
        # --- Sanity check ---
        if not model_filename.endswith(".model"):
            raise ValueError(f"Expected a .model file, got: {model_filename}")
        if not os.path.exists(model_full_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        # read the model file
        merges: dict[tuple[int, int], int] = {}
        special_tokens: dict[str, int] = {}
        idx: int = 256
        with open(model_full_path, encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            if version != "minbpe v1":
                raise ValueError(f"Unsupported model version: {version}")
            # read the pattern
            self.pattern = f.readline().strip()
            # read the number of special tokens
            num_special_str: str = f.readline().strip()
            num_special: int = int(num_special_str)
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                # Each line should contain two token IDs separated by a space
                left_token_str, right_token_str = line.strip().split()
                # Convert both IDs to integers
                left_token = int(left_token_str)
                right_token = int(right_token_str)
                # Register this merge pair with a new token ID
                merges[(left_token, right_token)] = idx
                idx += 1
        self.token_merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
