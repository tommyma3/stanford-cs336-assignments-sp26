from cs336_basics.bpe_tokenizer import (
    find_chunk_boundaries as find_chunk_boundaries_bpe,
)
from cs336_basics.pretokenization_example import (
    find_chunk_boundaries as find_chunk_boundaries_reference,
)


def test_chunk_boundaries():
    file_name = "data/TinyStoriesV2-GPT4-valid.txt"

    special_token = b"<|endoftext|>"
    special_token_list = [special_token]

    with open(file_name, "rb") as f:
        actual = find_chunk_boundaries_bpe(
            f, 4, special_token_list
        )

        f.seek(0)

        expected = find_chunk_boundaries_reference(
            f, 4, special_token
        )

    assert actual == expected