import os
import requests
from tqdm import tqdm
from model_utils import SimpleTokenizer

def download_file(url, filename):
    if os.path.exists(filename):
        print(f"File {filename} already exists, skipping download.")
        return
    
    print(f"Downloading {url} to {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    
    if total_size != 0 and t.n != total_size:
        print("ERROR, something went wrong")
    else:
        print(f"Successfully downloaded {filename}")

def prepare_tinystories():
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
    raw_path = "tinystories.txt"
    vocab_path = "vocab.txt"
    
    # Download
    download_file(url, raw_path)
    
    # Read a sample to build vocab or use a fixed set of characters
    # TinyStories is large, so we'll read a chunk to ensure we have all common chars
    print("Building vocabulary...")
    with open(raw_path, 'r', encoding='utf-8') as f:
        # Read first 10MB to get characters
        sample_data = f.read(10 * 1024 * 1024)
        
    tokenizer = SimpleTokenizer(text=sample_data)
    tokenizer.save_vocab(vocab_path)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Vocab saved to {vocab_path}")
    
    print("Dataset preparation complete.")

if __name__ == "__main__":
    prepare_tinystories()
