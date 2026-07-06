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
    frequency_table: dict[tuple[bytes, ...], int],
    special_tokens: list[str]
) -> None:
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    special_pattern = "|".join(re.escape(token) for token in special_tokens)
    parts = re.split(special_pattern, chunk)

    for part in parts:
        for match in re.finditer(PAT, part):
            token = match.group().encode("utf-8")
            token_tuple = tuple(bytes([b]) for b in token)
            frequency_table[token_tuple] = frequency_table.get(token_tuple, 0) + 1   


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> dict[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "rb") as f:
        num_processes = 4
        special_tokens_bytes = [s.encode("utf-8") for s in special_tokens]
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens_bytes)
        frequency_table = {}

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            pretokenize(chunk, frequency_table, special_tokens)
        
        

          
    
    return None


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 10000, special_tokens)