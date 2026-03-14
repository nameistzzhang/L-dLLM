from datasets import load_from_disk

ARROW_DIR = "/scratch/aszalay1/tianze/cpt_data/3b_pyarrow/"

print(f"Loading Arrow dataset from {ARROW_DIR}...")
dataset = load_from_disk(ARROW_DIR)

# 每条数据的固定长度
chunk_len = 4096
total_chunks = len(dataset)
total_tokens = total_chunks * chunk_len

print("="*40)
print(f"✅ Total Examples (Chunks): {total_chunks:,}")
print(f"✅ Total Tokens (incl. Padding): {total_tokens:,} ({total_tokens / 1e9:.2f} Billion)")
print("="*40)