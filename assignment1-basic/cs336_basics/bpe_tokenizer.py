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
    
    while len(vocab) < vocab_size and token_pair_table:
        max_pair = min(token_pair_table, key=lambda pair: (-token_pair_table[pair], pair))
        max_pair_first, max_pair_second = max_pair

        new_vocab_bytes = vocab[max_pair_first] + vocab[max_pair_second]
        new_token_id = len(vocab)
        vocab[new_token_id] = new_vocab_bytes
        merges.append((vocab[max_pair_first], vocab[max_pair_second]))

        del token_pair_table[max_pair]

        new_frequency_table = {}

        for word, freq in frequency_table.items():
            new_word = []
            i = 0

            while i < len(word):
                if i < len(word) - 1 and word[i] == max_pair_first and word[i + 1] == max_pair_second:
                    if len(new_word) > 0:
                        old_left_pair = (new_word[-1], max_pair_first)
                        token_pair_table[old_left_pair] = token_pair_table.get(old_left_pair, 0) - freq
                        if token_pair_table[old_left_pair] <= 0:
                            del token_pair_table[old_left_pair]
                    
                    if i + 2 < len(word):
                        old_right_pair = (max_pair_second, word[i + 2])
                        token_pair_table[old_right_pair] = token_pair_table.get(old_right_pair, 0) - freq
                        if token_pair_table[old_right_pair] <= 0:
                            del token_pair_table[old_right_pair]

                    new_word.append(new_token_id)

                    if len(new_word) > 1:
                        new_left_pair = (new_word[-2], new_token_id)
                        token_pair_table[new_left_pair] = token_pair_table.get(new_left_pair, 0) + freq
                    
                    if i + 2 < len(word):
                        new_right_pair = (new_token_id, word[i + 2])
                        token_pair_table[new_right_pair] = token_pair_table.get(new_right_pair, 0) + freq
                    
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
           
            new_word_tuple = tuple(new_word)
            new_frequency_table[new_word_tuple] = new_frequency_table.get(new_word_tuple, 0) + freq
        
        frequency_table.clear()
        frequency_table.update(new_frequency_table)
            

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