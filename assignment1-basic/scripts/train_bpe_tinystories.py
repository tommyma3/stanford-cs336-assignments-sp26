import pickle
import os  # Added for path and directory handling
from cs336_basics.bpe_tokenizer import train_bpe

OUTPUT_DIR = "artifacts/tinystories_tokenizer"

def main():
    special_tokens = ["<|endoftext|>"]
    vocab_size = 10000
    tiny_story_path = "data/TinyStories-train.txt"

    # 1. Train the BPE tokenizer
    vocab, merges = train_bpe(input_path=tiny_story_path, vocab_size=vocab_size, special_tokens=special_tokens)

    # 2. Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. Use os.path.join for correct path concatenation
    vocab_path = os.path.join(OUTPUT_DIR, "vocab.pkl")
    merges_path = os.path.join(OUTPUT_DIR, "merges.pkl")

    # 4. Open files in "wb" (write binary) mode for pickle
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
        
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    print(f"Tokenizer saved successfully to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()