import os
from typing import BinaryIO
import regex as re

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
    
    special_pattern = "|".join(re.escape(token) for token in special_tokens)
    parts = re.split(special_pattern, chunk)

    for part in parts:
        for match in re.finditer(PAT, part):
            token = match.group().encode("utf-8")
            token_tuple = tuple(int(b) for b in token)
            frequency_table[token_tuple] = frequency_table.get(token_tuple, 0) + 1   

def bpe_merge(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    frequency_table: dict[tuple[int, ...], int],
    vocab_size: int
) -> None:
    if len(vocab) >= vocab_size:
        return
    token_pair_table = {}

    for word, frequency in frequency_table.items():
        for first, second in zip(word[:-1], word[1:]):
            token_pair_table[(first, second)] = token_pair_table.get((first, second), 0) + frequency
    
    bpe_merge_helper(vocab, merges, frequency_table, vocab_size, token_pair_table)

    
def bpe_merge_helper(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    frequency_table: dict[tuple[int, ...], int],
    vocab_size: int,
    token_pair_table: dict[tuple[int, int], int]
) -> None:
    max_pair_first, max_pair_second = max(token_pair_table, key=lambda x: x[1])
    token_pair_table.pop((max_pair_first, max_pair_second))
    new_vocab_bytes = vocab[max_pair_first] + vocab[max_pair_second]
    merges.append((vocab[max_pair_first], vocab[max_pair_second]))
    vocab[len(vocab)] = new_vocab_bytes

    if len(vocab) >= vocab_size:
        return
    
    for word, frequency in frequency_table.items():
        for i in range(len(word) - 1):
            if (word[i] == max_pair_first and word[i + 1] == max_pair_second):
                if i > 0:
                    token_pair_table[(word[i - 1], max_pair_first)] -= frequency
                    token_pair_table[(word[i - 1], len(vocab))] = token_pair_table.get((word[i - 1], len(vocab)), 0) + frequency
                if i < len(word) - 2:
                    token_pair_table[(max_pair_second, word[i + 2])] -= frequency
                    token_pair_table[(len(vocab), word[i + 2])] = token_pair_table.get((len(vocab), word[i + 2]), 0) + frequency
    
    bpe_merge_helper(vocab, merges, frequency_table, vocab_size, token_pair_table)
    

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
          
    
    return None


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 10000, special_tokens)