import os
from typing import BinaryIO

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
            

def pretokenization(
    chunk: str,
    desired_num_chunks: int,
    special_tokens: list[str]
) -> dict[bytes, int]:
    return None
    


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> dict[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "rb") as f:
        special_tokens_bytes = [s.encode("utf-8") for s in special_tokens]
          
    
    return None