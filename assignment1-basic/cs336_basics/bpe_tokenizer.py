import os
from typing import BinaryIO
import regex as re
from collections import defaultdict, Counter

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    special_tokens: list[bytes],
) -> list[int]:
    
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk_size
            for s in special_tokens:
                position = mini_chunk.find(s)
                if position >= 0 and position < found_at:
                    found_at = position
            if found_at != mini_chunk_size:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    
    return sorted(set(chunk_boundaries))
            

def pretokenize(
    chunk: str,
    frequency_table: dict[tuple[int, ...], int],
    special_tokens: list[str]
) -> None:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    if special_tokens:
        special_pattern = "|".join(re.escape(token) for token in special_tokens)
        parts = re.split(special_pattern, chunk)
    else:
        parts = [chunk]

    for part in parts:
        for match in re.finditer(PAT, part):
            token = match.group().encode("utf-8")
            token_tuple = tuple(token)
            frequency_table[token_tuple] = frequency_table.get(token_tuple, 0) + 1


def get_pair_counter(word: tuple[int, ...]) -> Counter[tuple[int, int]]:
    return Counter(zip(word[:-1], word[1:]))


def merge_word(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    out = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def bpe_merge(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    frequency_table: dict[tuple[int, ...], int],
    vocab_size: int,
) -> None:
    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)

    for word, freq in frequency_table.items():
        pair_counter = get_pair_counter(word)
        for pair, count in pair_counter.items():
            pair_counts[pair] += count * freq
            pair_to_words[pair].add(word)

    while len(vocab) < vocab_size and pair_counts:
        best_pair = max(
            pair_counts,
            key=lambda pair: (
                pair_counts[pair],
                vocab[pair[0]],
                vocab[pair[1]],
            ),
        )

        new_id = len(vocab)
        first, second = best_pair
        vocab[new_id] = vocab[first] + vocab[second]
        merges.append((vocab[first], vocab[second]))

        affected_words = list(pair_to_words[best_pair])

        for old_word in affected_words:
            if old_word not in frequency_table:
                continue

            freq = frequency_table.pop(old_word)

            old_pair_counter = get_pair_counter(old_word)

            for pair, count in old_pair_counter.items():
                pair_counts[pair] -= count * freq
                pair_to_words[pair].discard(old_word)

                if pair_counts[pair] == 0:
                    del pair_counts[pair]
                    del pair_to_words[pair]

            new_word = merge_word(old_word, best_pair, new_id)

            old_existing_freq = frequency_table.get(new_word, 0)
            frequency_table[new_word] = old_existing_freq + freq

            new_pair_counter = get_pair_counter(new_word)

            for pair, count in new_pair_counter.items():
                pair_counts[pair] += count * freq
                pair_to_words[pair].add(new_word)
            

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "rb") as f:
        num_processes = 4
        special_tokens_bytes = [s.encode("utf-8") for s in special_tokens]
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens_bytes)
        frequency_table = {}

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            pretokenize(chunk, frequency_table, special_tokens)
        
        vocab = {}
        for t in range(256):
            vocab[t] = bytes([t])
        for i in range(len(special_tokens)):
            vocab[i + 256] = special_tokens[i].encode("utf-8")

        merges = []
        bpe_merge(vocab, merges, frequency_table, vocab_size)  

        return vocab, merges
    
    return None


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 10000, special_tokens)